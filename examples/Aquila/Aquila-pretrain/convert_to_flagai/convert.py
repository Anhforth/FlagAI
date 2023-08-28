import torch
import shutil
import os
# import sys;sys.path.append("/data2/yzd/flagai-internal_save/")
from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.llama import llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint
from flagai.model.aquila_model import AQUILAModel
# from transpose_weight import transpose_gpt2_to_llama_weight

# config_file = "config.json"
model_name = "aquila-7b"

# 1. 用flash atten加载flash attn 模型
# checkpoint_path = '/data2/ldwang/checkpoints/model_hub/'
# model_name = sys.argv[1]
# version = sys.argv[2]
model_name = model_name.lower()
checkpoint_path = './checkpoints_store/flash_attn_checkpoints/'
config_file = checkpoint_path+model_name+'/config.json'

out_dir = './checkpoints_store/converted_models/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(out_dir+model_name):
    os.mkdir(out_dir+model_name,)

config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoint_path, model_name))
config.vocab_size = 100008
config.use_cache = True
config.attn_pdrop = 0.0
config.resid_pdrop = 0.0
config.fused_bias_fc = False
config.layer_norm_epsilon = 1e-5

config.fused_mlp = False  # We don't have fused GatedMLP yet
config.fused_dropout_add_ln = False
config.residual_in_fp32 = False
config.bmt = False
config.prenorm = True 
config.use_flash_attn = True

print(f'config {config}')

dtype = torch.float16
device="cuda"
model = GPTLMHeadModel(config, device=device, dtype=dtype)
ckpt = torch.load(checkpoint_path+model_name+'/pytorch_model.bin', map_location=device)
model.load_state_dict(ckpt, strict=True)
# model.transformer.bmt_replace()
sd = model.state_dict()
torch.save(sd, './checkpoints_store/model.pt')
model.eval()


def transform_flash_to_flagai(ckpt):
    tgt_ckpt = {}
    tgt_ckpt["tok_embeddings.weight"] =  ckpt.pop("transformer.embeddings.word_embeddings.weight")
    tgt_ckpt["output.weight"] =  ckpt.pop("lm_head.weight")
    tgt_ckpt["norm.weight"] = ckpt.pop("transformer.ln_f.weight")

    for l in range(config.n_layer):
        #tgt_ckpt[f"layers.{l}.attention.rotary_emb.inv_freq"] = ckpt.pop(f"transformer.layers.{l}.mixer.rotary_emb.inv_freq")
    
        # attention
        Wqkv = ckpt.pop(f'transformer.layers.{l}.mixer.Wqkv.weight') 
        # tgt_ckpt[f'layers.{l}.attention.Wqkv.weight']= Wqkv
        split_size = Wqkv.size()[0]//3
        Wq, Wk, Wv= torch.split(Wqkv,split_size)
        tgt_ckpt[f'layers.{l}.attention.wq.weight'] = Wq
        tgt_ckpt[f'layers.{l}.attention.wk.weight'] = Wk
        tgt_ckpt[f'layers.{l}.attention.wv.weight'] = Wv
        tgt_ckpt[f'layers.{l}.attention.wo.weight']=ckpt.pop(f'transformer.layers.{l}.mixer.out_proj.weight')

        # feedforward
        W31 = ckpt.pop(f'transformer.layers.{l}.mlp.fc1.weight')
        #tgt_ckpt[f'layers.{l}.feed_forward.fc1.weight'] = W31
        split_size = W31.size()[0]//2
        W3, W1= torch.split(W31,split_size)
        tgt_ckpt[f'layers.{l}.feed_forward.w1.weight'] = W1
        tgt_ckpt[f'layers.{l}.feed_forward.w3.weight'] = W3
        tgt_ckpt[f'layers.{l}.feed_forward.w2.weight'] = ckpt.pop(f'transformer.layers.{l}.mlp.fc2.weight')

        # layernorm
        tgt_ckpt[f"layers.{l}.attention_norm.weight"] = ckpt.pop(f'transformer.layers.{l}.norm1.weight') 
        tgt_ckpt[f"layers.{l}.ffn_norm.weight"] = ckpt.pop(f'transformer.layers.{l}.norm2.weight')
    return tgt_ckpt


# 加载flagai的模型
model_flagai = AQUILAModel.init_from_json(config_file=config_file)
model_flagai.eval()
checkpoint_path = "./checkpoints_store/model.pt"
ckpt = torch.load(checkpoint_path, map_location=device)
model_flagai.half()

# 转换模型
model_flagai.load_state_dict(transform_flash_to_flagai(ckpt), strict=True)

# 保存模型
torch.save(model_flagai.state_dict(), out_dir+model_name+'/pytorch_model.bin')



