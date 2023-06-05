##

export CELL_LINE=K562
export OUTPUT_DIR=./result
export MODEL=tcn
export FAST_FILE_PATH=./on-example.fasta
echo --------------------------------start_predicting--------------------------------
echo "input path will be ./on-example.fasta"
echo "$MODEL was chosen"
echo "$CELL_LINE cellline was chosen"
python3 predictor.py\
    --cellline $CELL_LINE\
    --input $FAST_FILE_PATH\
    --model $MODEL\
    --output_dir $OUTPUT_DIR 
echo --------------------------------finish_predicting--------------------------------
echo "output file will be $OUTPUT_DIR"
