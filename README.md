# For on-target prediction 

## Step one 

    cd ./predictors/ON-target

## Step three 

    vim run.sh

    export CELL_LINE=K562
    export OUTPUT_DIR=./result
    export MODEL=tcn
    export FILE_PATH=./on-example.fasta
    echo --------------------------------start_predicting--------------------------------
    echo "input path will be $FILE_PATH"
    echo "$MODEL was chosen"
    echo "$CELL_LINE cellline was chosen"
    python3 predictor.py\
        --cellline $CELL_LINE\
        --input $FILE_PATH\
        --model $MODEL\
        --output_dir $OUTPUT_DIR 
    echo --------------------------------finish_predicting--------------------------------
    echo "output file will be $OUTPUT_DIR"

parameter:  
CELL_LINE: choose cellline (K562 or Jurkat)  
OUTPUT_DIR: output
FILE_PATH: fasta file used for prediction  

## Step three 
    bash run.sh

# For off-target prediction 

## Step one 

    cd ./predictors/OFF-target

## Step two 
    vim run.sh 

    export CELL_LINE=K562
    export OUTPUT_DIR=./result
    export MODEL=mlp
    export FILE_PATH=./K562_sample.txt
    echo --------------------------------start_predicting--------------------------------
    echo "input path will be $FILE_PATH"
    echo "$MODEL was chosen"
    echo "$CELL_LINE cellline was chosen"
    python3 predictor.py\
        --cellline $CELL_LINE\
        --input $FILE_PATH\
        --model $MODEL\
        --output_dir $OUTPUT_DIR 
    echo --------------------------------finish_predicting--------------------------------
    echo "output file will be $OUTPUT_DIR"

parameter:  
CELL_LINE: choose cellline (K562 or Jurkat)  
OUTPUT_DIR: output
FILE_PATH: txt file used for prediction  

## Step three

    bash run.sh
