export PYTHONPATH=/data2/yzd/FlagAI

PREPROCESS_DATA_TOOL=$PYTHONPATH/flagai/data/dataset/indexed_dataset/preprocess_data_args.py
TOKENIZER_DIR=/data2/yzd/FlagAI/examples/indexed_dataset/
TOKENIZER_NAME=gpt2-base-en

INPUT_FILE=/data2/yzd/FlagAI/examples/indexed_dataset/data/zhihu.jsonl
FULL_OUTPUT_PREFIX=./data/zhihu
echo $TOKENIZER_NAME
python $PREPROCESS_DATA_TOOL --input $INPUT_FILE --output-prefix $FULL_OUTPUT_PREFIX \
    --workers 4 --chunk-size 256 \
    --model-name $TOKENIZER_NAME --model-dir $TOKENIZER_DIR
