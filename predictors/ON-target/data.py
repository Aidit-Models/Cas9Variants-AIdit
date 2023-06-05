from Bio import SeqIO
import numpy as np
import pandas as pd 


# 用 one-hot 的编码方式来决定我们的输入
def one_hot(input):
    """ convert one sequence input to one-hot style
    Args:
        Input: one seuqence example:ATGC (dtype:str)
    
    Return: 
        one-hot output (dtype:array)
    """
    data=[]
    for i in range(len(input)):
        if input[i] ==  'A':
            output_data = [1,0,0,0]
        elif input[i] == 'T':
            output_data = [0,1,0,0]
        elif input[i] == 'G':
            output_data = [0,0,1,0]
        elif input[i] == 'C':
            output_data = [0,0,0,1]
        data.append(output_data)
    return np.array(data)

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

# 改变输入数据的类型
def to_array(input):
    """convert  input from list to array(from 1 to 4)
    Args:
        Input:  (dtype:array)
    
    Return: 
        output (dtype:array)
    """
    return np.array(input.to_list())

# 通过输入的序列信息或者文件信息找到所有可以作为输入的 63bp 的输入
def find_all(input,file=True):
    sequence = []
    if file:
        if 'fasta' in input:
            records = SeqIO.parse(input,'fasta')
            for record in records:
                s = str(record.seq)
                pattern = 'GG'
                positions = [i for i in range(len(s)) if s.find(pattern, i) == i]
                seq = [s[pos-40:pos+23] for pos in positions if pos >=40 and pos+23 <= len(s)]
                sequence = sequence + seq 
        elif 'txt' in input:
            with open(input,'r') as f:
                for line in f:
                    s = line
                    pattern = 'GG'
                    positions = [i for i in range(len(s)) if s.find(pattern, i) == i]
                    seq = [s[pos-40:pos+23] for pos in positions if pos >=40 and pos+23 <= len(s)]
                    sequence = sequence + seq 
        else:
            print('不支持该文件格式')
    else:
        s = input
        pattern = 'GG'
        positions = [i for i in range(len(s)) if s.find(pattern, i) == i]
        seq = [s[pos-40:pos+23] for pos in positions if pos >=40 and pos+23 <= len(s)]
        sequence = sequence + seq
    return sequence

# 通过数据创造一个一维的数据向量作为on-target输入
def creat_data_for_1d_input(token_type,dataframe):
    if token_type == 'embedding':
        dataprocess = emb_number
        out_shape = (-1,63)
    elif token_type == 'one_hot':
        dataprocess = one_hot
        out_shape = (-1,63*4)

    dataframe['input'] = dataframe['sequence'].apply(lambda x:dataprocess(x))
    input = to_array(dataframe['input'])
    return input.reshape(out_shape)

# 通过数据创造一个二维的数据向量作为on-target输入
def creat_data_for_2d_input(token_type,dataframe):
    if token_type == 'embedding':
        dataprocess = emb_number
        out_shape = (-1,63,1)
    elif token_type == 'one-hot':
        dataprocess = one_hot
        out_shape = (-1,63,4)
    dataframe['input'] = dataframe['sequence'].apply(lambda x:dataprocess(x))
    input = to_array(dataframe['input'])
    return input.reshape(out_shape)
