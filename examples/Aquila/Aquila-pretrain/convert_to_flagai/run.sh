export MODEL_NAME="aquila-7b"  # "aquilachat-7b"
export LOC="/data2/yzd/pytorch_model.bin"  # 需要修改
 
mkdir checkpoints_store/flash_attn_checkpoints/${MODEL_NAME}

directory="checkpoints_store/flash_attn_checkpoints/${MODEL_NAME}"
# 判断目录是否存在
if [ ! -d "$directory" ]; then
    # 如果目录不存在，则创建目录
    mkdir -p "$directory"
    echo "目录已创建：$directory"
else
    echo "目录已存在：$directory"
fi
cp checkpoints_store/model_files/* checkpoints_store/flash_attn_checkpoints/${MODEL_NAME}



#sshpass -p 'p5dkf@v2QvSGm@bW%aTc' scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r ${LOC} ./

cp $LOC checkpoints_store/flash_attn_checkpoints/${MODEL_NAME}

python convert.py ${MODEL_NAME} 
cp checkpoints_store/model_files/* checkpoints_store/converted_models/${MODEL_NAME}
cp -r checkpoints_store/converted_models/${MODEL_NAME} ../checkpoints_in/