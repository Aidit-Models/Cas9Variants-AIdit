import os 
import pandas as pd 
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from metric_loss import *

# 生成路径文件
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

# 将 one-hot 作为基本编码信息
def one_hot(input,one_hot_dict):
    data = list(map(lambda x:one_hot_dict[x], input))
    return np.array(data)

def get_position(wt,ot):
    ## 基于序列信息获得 mutation position
    position=[]
    for i in range(len(wt)):
        if wt[i] == ot[i]:
            position.append(0)
        else:
            position.append(1)
    return position


# 用embedding 的编码方式来决定我们的输入
def emb_number(input):
    """convert one sequence input to tokens(from 1 to 4)
    Args:
        Input: one seuqence example:ATGC (dtype:str)
    
    Return: 
        the tokens of one input (dtype:array)
    """
    data=[]
    for i in range(len(input)):
        if input[i] ==  'A':
            output_data = 0
        elif input[i] == 'T':
            output_data = 1
        elif input[i] == 'G':
            output_data = 2
        elif input[i] == 'C':
            output_data = 3
        data.append(output_data)
    return np.array(data)

# 通过数据创造一个一维的数据向量作为on-target输入
def creat_data_for_1d_input(token_type,dataframe):
    if token_type == 'embedding':
        dataprocess = emb_number
        out_shape = (-1,63)

    dataframe['input'] = dataframe['sequence'].apply(lambda x:dataprocess(x))
    input = to_array(dataframe['input'])
    return input.reshape(out_shape)

# 改变输入数据的类型
def to_array(input):
    """convert  input from list to array(from 1 to 4)
    Args:
        Input:  (dtype:array)
    
    Return: 
        output (dtype:array)
    """
    return np.array(input.to_list())

# 将label数据转换成我们需要的shape
def creat_label (dataframe):
    return to_array(dataframe['off']).reshape(-1,1)

# 获取使用 on-target indel 作为特征
def pred_on_target_indel (input,model_library,cellline):
    data = pd.DataFrame(input,columns=['sequence'])
    model_selection = f'{model_library}/{cellline}_best_model/rnn_best.h5'
    model = tf.keras.models.load_model(model_selection,custom_objects={'focal_loss':focal_loss,'get_spearmanr':get_spearmanr})
    embedding_1d_input = creat_data_for_1d_input('embedding',data)
    output = model.predict(embedding_1d_input)
    return output

# off target 所有特征生成
def off_target_process(dataframe,model_library,cellline):
    ## 预测on-target indel 作为特征之一
    wt_list = list(set(dataframe['wt_63_seq']))
    ontarget_ = pred_on_target_indel(wt_list,model_library,cellline)
    ontarget = pd.DataFrame(wt_list)
    ontarget.columns = ['wt_63_seq']
    ontarget['on target indel'] = ontarget_
    ## 将我们需要使用的序列信息，位置信息以及on-target indel 作为输入进行处理
    joint_seq = []
    position = []
    on_target_indel = []
    one_hot_dict  = dict({'A':[1,0,0,0],'T':[0,1,0,0],'G':[0,0,1,0],'C':[0,0,0,1]})
    on_dict = dict(zip(ontarget['wt_63_seq'],ontarget['on target indel']))
    on = list(dataframe['wt_63_seq'])
    off = list(dataframe['ot_63_seq'])
    for on_,off_ in zip(on,off):
        on_key = on_
        on_ = on_[20:43]
        off_ = off_[20:43]

        if 'N' not in off_:
            joint_seq_ = one_hot(on_+off_,one_hot_dict)
            joint_seq.append(joint_seq_,)
            position_ = get_position(on_,off_)
            position.append(position_)
            on_target_indel_= on_dict[on_key]
            on_target_indel.append(on_target_indel_)
    dataframe['on_target_indel'] = on_target_indel
    dataframe['position'] = position
    dataframe['joint_seq'] = joint_seq
    return dataframe

# 创建一个专门给MLP模型的输入
def creat_off_input_for_mlp(dataframe):
    return [to_array(dataframe['joint_seq']),to_array(dataframe['position']),to_array(dataframe['on_target_indel'])]

# 创建一个专门给机器学习使用的输入
def creat_off_input_for_ml(dataframe):
    train_input_1 = np.array(dataframe['joint_seq'].to_list()).reshape(-1,46*4)
    train_input_2 = np.array(dataframe['position'].to_list()).reshape(-1,23)
    train_input_3 = np.array(dataframe['on_target_indel'].to_list()).reshape(-1,1)
    train_x = np.hstack([train_input_1,train_input_2,train_input_3])
    return train_x






