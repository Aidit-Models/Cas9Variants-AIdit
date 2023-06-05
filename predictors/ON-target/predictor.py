from data import *
from metric_loss import *
import tensorflow as tf
import argparse
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
    ## input file or input sequence 
    parser.add_argument(
        "--input",
        type=str,
        help="path of data for training and testing or just sequence",
    )
    ## choose model
    parser.add_argument(
        "--model",
        type=str,
        help="choose the model you would like to use (linear, xgb,lgb, rnn,tcn)",
    )
    ## on-target model library
    parser.add_argument(
        "--model_library",
        type=str,
        default='.',
        help="model_library path",
    )
    ## save path
    parser.add_argument(
        "--output_dir",
        type=str,
        default='.',
        help="output path",
    )
    args = parser.parse_args()
    
    ##check input type
    if is_valid_path(args.input):
        data = find_all(args.input,file=True)
    else:
        data = find_all(args.input,file=False)

    ##create folder 
    mkdir(args.output_dir)
    ## prepare data
    data = pd.DataFrame(data)
    data.columns = ['sequence']
    one_hot_1d_input = creat_data_for_1d_input('one_hot',data)
    embedding_1d_input = creat_data_for_1d_input('embedding',data)

    ## choose model
    if 'ridge' in args.model or 'xgb' in args.model or 'linear' in args.model or "lgb" in args.model:
        model_selection = f'{args.model_library}/{args.cellline}_best_model/{args.model}_best.sav'
        ml = joblib.load(model_selection)
        output = ml.predict(one_hot_1d_input)
        data['pred'] = output
        data = data[['sequence','pred']]
        data.to_csv(args.output_dir+'/pred.csv',index=False)
    else:
        model_selection = f'{args.model_library}/{args.cellline}_best_model/{args.model}_best.h5'
        model = tf.keras.models.load_model(model_selection,custom_objects={'focal_loss':focal_loss,'get_spearmanr':get_spearmanr})
        output = model.predict(embedding_1d_input)
        data['pred'] = output
        data = data[['sequence','pred']]
        data.to_csv(args.output_dir+'/pred.csv',index=False)

if __name__=='__main__':
    pred()
