from data import *
from metric_loss import *
import tensorflow as tf
import argparse
import joblib
import os



# 检查输入是否为路径
def is_valid_path(path):
    return os.path.isfile(path) or os.path.isdir(path)

# 生成路径文件
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

# 定义预测函数
def pred():
    parser = argparse.ArgumentParser()
    ## 输入细胞系
    parser.add_argument(
        "--cellline",
        type=str,
        help="choose K562 cellline or Jurakt cellline ",
    )
    ## 输入文件路径或者序列信息
    parser.add_argument(
        "--input",
        type=str,
        help="path of data for training and testing or just sequence",
    )
    ## 输入想要选择的模型
    parser.add_argument(
        "--model",
        type=str,
        help="choose the model you would like to use (linear, xgb,lgb, rnn,tcn)",
    )
    ## 输入模型的保存路进
    parser.add_argument(
        "--model_library",
        type=str,
        default='.',
        help="model_library path",
    )
    ## 输出结果路径
    parser.add_argument(
        "--output_dir",
        type=str,
        default='.',
        help="output path",
    )
    args = parser.parse_args()
    
    ##检查输入是否为路径或者单纯的字符号
    if is_valid_path(args.input):
        data = find_all(args.input,file=True)
    else:
        data = find_all(args.input,file=False)

    ##检查输出路径，如果不存在并创建
    mkdir(args.output_dir)
    ## 对数据进行处理并做成我们想要的输出
    data = pd.DataFrame(data)
    data.columns = ['sequence']
    one_hot_1d_input = creat_data_for_1d_input('one_hot',data)
    embedding_1d_input = creat_data_for_1d_input('embedding',data)

    ## 选择我们想要的模型进行预测
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
