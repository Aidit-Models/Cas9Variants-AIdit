from data import *
from metric_loss import *
import tensorflow as tf
import argparse
import logging
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

# 定义预测使用的函数
def pred():
    parser = argparse.ArgumentParser()
    ## 输入的细胞系
    parser.add_argument(
        "--cellline",
        type=str,
        help="choose K562 cellline or Jurakt cellline ",
    )
    ## 输入的文件或者序列信息
    parser.add_argument(
        "--input",
        type=str,
        help="path of data for training and testing or just sequence",
    )
    ## 输入模型
    parser.add_argument(
        "--model",
        type=str,
        help="choose the model you would like to use ( lgb, ridge,xgb and  mlp)",
    )
    ## 输入模型保存的库
    parser.add_argument(
        "--model_library",
        type=str,
        default='.',
        help="model_library path",
    )
    ## 输入on-target 模型库
    parser.add_argument(
        "--on_target_model_path",
        type=str,
        default='../ON-target',
        help="path of on target library ",
    )
    ## 输出结果路径
    parser.add_argument(
        "--output_dir",
        type=str,
        default='.',
        help="output path",
    )
    args = parser.parse_args()

    ## 检查是否是我们需要的路径
    if is_valid_path(args.input):
        if 'txt' in args.input:
            data = pd.read_table(args.input,header=None)
            data.columns = ['wt_63_seq','ot_63_seq']
        else:
            print('请输入txt格式')
    else:
        print('请输入txt格式')
        
    ##检查输出路径，如果不存在并创建
    mkdir(args.output_dir)
    
    ## 准备数据
    data = off_target_process(data,args.on_target_model_path,args.cellline)
    input_for_mlp = creat_off_input_for_mlp(data)
    input_for_ml = creat_off_input_for_ml(data)

    ##选择模型 进入不同的处理流程
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
