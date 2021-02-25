""" MMD model components.

Author: Nathaniel Andre
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input,Dense,Dropout,Activation,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")


def get_mmd_unit(mat1,mat2,sigma=1,batch_size=50):
    """ calculates MMD components
    """
    mat1 = tf.expand_dims(mat1,axis=0) # done for tf broadcasting
    mat2 = tf.expand_dims(mat2,axis=1)
    diff = tf.reshape(tf.subtract(mat1,mat2),[batch_size,-1]) # difference between all rows in mat1 and mat2, stacked
    squared_euclid_distance = tf.reduce_sum(tf.square(diff),axis=1)
    kernel_sum = tf.reduce_mean(tf.exp(-squared_euclid_distance/sigma)) # calculating RBF kernel
    return kernel_sum


def mmd_model(input_dim):
    """ model implementation
    """
    x = Input(shape=(input_dim))
    h = Dense(256,activation=None)(x)
    h_2 = Activation('relu')(h)
    out = Dense(1,activation='sigmoid')(h_2)
    
    model = Model(inputs=x,outputs=[out,h])
    return model


def mmd_loss_semi_supervised(source_y_subset,source_pred,target_y_subset,target_pred,target_h,source_h,mmd_lam):
    """ loss function with MMD    
    args:
        mmd_lam: controls the contribution of MMD loss component
    """
    # task-specific loss:
    source_task_loss = BinaryCrossentropy()(source_y_subset,source_pred) # automatic avg over batch
    target_task_loss = BinaryCrossentropy()(target_y_subset,target_pred)
    task_loss = 1/2*(source_task_loss+target_task_loss) # semi-supervised
    # MMD loss:
    mmd_loss = get_mmd_unit(target_h,target_h)+get_mmd_unit(source_h,source_h)-2*get_mmd_unit(target_h,source_h)

    total_loss = task_loss+(mmd_lam*mmd_loss)
    return total_loss


def mmd_loss_unsupervised(source_y_subset,source_pred,target_y_subset,target_pred,target_h,source_h,mmd_lam):
    """ loss function with MMD    
    args:
        mmd_lam: controls the contribution of MMD loss component
    """
    # task-specific loss:
    source_task_loss = BinaryCrossentropy()(source_y_subset,source_pred) # automatic avg over batch
    task_loss = source_task_loss # unsupervised
    # MMD loss:
    mmd_loss = get_mmd_unit(target_h,target_h)+get_mmd_unit(source_h,source_h)-2*get_mmd_unit(target_h,source_h)

    total_loss = task_loss+(mmd_lam*mmd_loss)
    return total_loss


def get_single_mmd_model_simulation(model,source_x,source_y,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size,model_type,optimizer=Adam(lr=0.003),use_cutoff=False,cutoff=0.9):
    """ Training a single instance of the MMD model
    """
    @tf.function # this is required to be a nested function
    def train_mmd(model,optimizer,source_x_subset,source_y_subset,target_x_subset,target_y_subset,loss_func,mmd_lam=1.0):
        """ used to train the model
        """
        with tf.GradientTape() as tape:
            source_pred,source_h = model(source_x_subset)
            target_pred,target_h = model(target_x_subset)
            loss = loss_func(source_y_subset,source_pred,target_y_subset,target_pred,target_h,source_h,mmd_lam)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    best_performance=(0,0,0,0) # keeping track of the best model epoch: (val_acc,val_bal-acc,test_acc,test_bal-acc)
    used_cutoff=False # bool: tracks whether the cutoff was used

    if model_type=="semi_supervised": # different loss functions depending on whether target-domain data is incorporated into task loss
        loss_func = mmd_loss_semi_supervised
    elif model_type=="unsupervised":
        loss_func = mmd_loss_unsupervised

    for epoch_i in range(n_epochs):
        for i in range(0,len(source_x),batch_size):
            source_x_train_subset = source_x[i:i+batch_size]
            source_y_train_subset = source_y[i:i+batch_size]
            target_x_train_subset = target_x[i%len(target_x):i%len(target_x)+batch_size] 
            target_y_train_subset = target_y[i%len(target_y):i%len(target_y)+batch_size]
            batch_loss = train_mmd(model,optimizer,source_x_train_subset,source_y_train_subset,target_x_train_subset,target_y_train_subset,loss_func)
        
        # getting the predictions for the val/test sets:
        target_val_pred,_ = model(target_x_val)
        target_val_pred = target_val_pred.numpy()
        target_val_pred[target_val_pred >= 0.5]=1 ; target_val_pred[target_val_pred < 0.5]=0
        
        target_test_pred,_ = model(target_x_test)
        target_test_pred = target_test_pred.numpy()
        target_test_pred[target_test_pred >= 0.5]=1 ; target_test_pred[target_test_pred < 0.5]=0
        
        # getting the acc/bal_acc metrics:
        val_acc,val_bal_acc = round(accuracy_score(target_y_val,target_val_pred),4),round(balanced_accuracy_score(target_y_val,target_val_pred),4)
        test_acc,test_bal_acc = round(accuracy_score(target_y_test,target_test_pred),4),round(balanced_accuracy_score(target_y_test,target_test_pred),4)
        
        if val_acc > best_performance[0]: # updating the best val performance for this run
            best_performance = (val_acc,val_bal_acc,test_acc,test_bal_acc)

        if use_cutoff:
            if val_acc>=cutoff: #and val_bal_acc>=cutoff:
                used_cutoff=True
                break
        
    return model,best_performance,used_cutoff
