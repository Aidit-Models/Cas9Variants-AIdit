from data import *
from metric_loss import *
import tensorflow as tf
import argparse
import logging
import joblib
import os



# check path 
def is_valid_path(path):
    return os.path.isfile(path) or os.path.isdir(path)

# create folder 
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

# main function
def pred():
    parser = argparse.ArgumentParser()
    ## cell line 
    parser.add_argument(
        "--cellline",
        type=str,
        help="choose K562 cellline or Jurakt cellline ",
    )
    ## input file
    parser.add_argument(
        "--input",
        type=str,
        help="path of data for training and testing or just sequence",
    )
    ## choose model
    parser.add_argument(
        "--model",
        type=str,
        help="choose the model you would like to use ( lgb, ridge,xgb and  mlp)",
    )
    ## off-target model library
    parser.add_argument(
        "--model_library",
        type=str,
        default='.',
        help="model_library path",
    )
    ## on-target model library 
    parser.add_argument(
        "--on_target_model_path",
        type=str,
        default='../ON-target',
        help="path of on target library ",
    )
    ## output path 
    parser.add_argument(
        "--output_dir",
        type=str,
        default='.',
        help="output path",
    )
    args = parser.parse_args()

    ## check file type
    if is_valid_path(args.input):
        if 'txt' in args.input:
            data = pd.read_table(args.input,header=None)
            data.columns = ['wt_63_seq','ot_63_seq']
        else:
            print('need txt file')
    else:
        print('need txt file')
        
    ## create folder 
    mkdir(args.output_dir)
    
    ## prepare data
    data = off_target_process(data,args.on_target_model_path,args.cellline)
    input_for_mlp = creat_off_input_for_mlp(data)
    input_for_ml = creat_off_input_for_ml(data)

    ##choose model
    if 'ridge' in args.model or 'xgb' in args.model or 'linear' in args.model or "lgb" in args.model:
        model_selection = f'{args.model_library}/{args.cellline}_best_model/{args.model}_best.sav'
        ml = joblib.load(model_selection)
        output = ml.predict(input_for_ml)
        data['pred'] = output
        data = data[['wt_63_seq','ot_63_seq','pred']]
        data.to_csv(args.output_dir+'/pred.csv',index=False)
    else:
        model_selection = f'{args.model_library}/{args.cellline}_best_model/{args.model}_best.h5'
        model = tf.keras.models.load_model(model_selection,custom_objects={'focal_loss':focal_loss,'get_spearmanr':get_personr})
        output = model.predict(input_for_mlp)
        data['pred'] = output
        data = data[['wt_63_seq','ot_63_seq','pred']]
        data.to_csv(args.output_dir+'/pred.csv',index=False)

if __name__=='__main__':
    pred()
