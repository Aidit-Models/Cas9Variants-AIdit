import tensorflow as tf 
from scipy.stats import spearmanr, pearsonr 



# metric 
def correlation(y_true, y_pred):
    import pandas as pd
    y_true = y_true.reshape(y_true.shape[0])
    y_pred = y_pred.reshape(y_true.shape[0])
    sp = pd.Series(y_pred).corr(pd.Series(y_true), method='spearman')
    pr = pd.Series(y_pred).corr(pd.Series(y_true), method='pearson')
    return sp, pr

# loss funciton
def focal_loss(y_true,y_pred):
    alpha= .25
    gamma=2.
    mae = tf.keras.losses.MeanAbsoluteError()
    l1_loss = mae(y_true,y_pred)
    loss = l1_loss*(2.*tf.math.sigmoid(alpha * l1_loss)-1.)**gamma
    loss = l1_loss + loss
    loss = tf.experimental.numpy.mean(loss)
    return loss


# metric which used during training 
"""check spearman and pearson during trainning """
def get_spearmanr(y_true, y_pred):  
    return tf.py_function(spearmanr, [y_true, y_pred], Tout = tf.float32)
    
def get_personr(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1,))
    y_pred = tf.reshape(y_pred, shape=(-1,))  
    return tf.py_function(pearsonr, [y_true, y_pred], Tout = tf.float32)
