""" CORAL model components.

Author: Nathaniel Andre
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input,Dense,Dropout,Activation,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")


def coral_component(mat1,mat2,d=256):
    """ calculates the CORAL loss component
    args:
        d: dimensionality of the input model hidden layer
    """
    mat1_cov = K.flatten(tfp.stats.covariance(mat1))
    mat2_cov = K.flatten(tfp.stats.covariance(mat2))
    squared_frobenius_distance = (1/(4*d**2))*tf.reduce_sum(tf.square(mat1_cov-mat2_cov))
    return squared_frobenius_distance


def coral_model(input_dim):
    """ model implementation
    """
    x = Input(shape=(input_dim))
    h = Dense(256,activation=None)(x)
    h_2 = Activation('relu')(h)
    out = Dense(1,activation='sigmoid')(h_2)
    
    model = Model(inputs=x,outputs=[out,h])
    return model


def coral_loss_semi_supervised(source_y_subset,source_pred,target_y_subset,target_pred,target_h,source_h,coral_lam=1.0):
    """ loss function with CORAL
    args:
        coral_lam: controls the effect of CORAL loss component
    """
    # task-specific loss:
    source_task_loss = BinaryCrossentropy()(source_y_subset,source_pred) # automatic avg over batch
    target_task_loss = BinaryCrossentropy()(target_y_subset,target_pred)
    task_loss = 1/2*(source_task_loss+target_task_loss) # semi-supervised
    # CORAL loss:
    coral_loss = coral_component(target_h,source_h)

    total_loss = task_loss+(coral_lam*coral_loss)
    return total_loss


def coral_loss_unsupervised(source_y_subset,source_pred,target_y_subset,target_pred,target_h,source_h,coral_lam=1.0):
    """ loss function with CORAL
    args:
        coral_lam: controls the effect of CORAL loss component
    """
    # task-specific loss:
    source_task_loss = BinaryCrossentropy()(source_y_subset,source_pred) # automatic avg over batch
    task_loss = source_task_loss # unsupervised
    # CORAL loss:
    coral_loss = coral_component(target_h,source_h)

    total_loss = task_loss+(coral_lam*coral_loss)
    return total_loss


def get_single_coral_model_simulation(model,source_x,source_y,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size,model_type,optimizer=Adam(lr=0.003),use_cutoff=False,cutoff=0.9):
    """ Training a single instance of the CORAL model
    """
    @tf.function # this is required to be a nested function
    def train_coral(model,optimizer,source_x_subset,source_y_subset,target_x_subset,target_y_subset,loss_func,coral_lam=1.0):
        """ used to train the model
        """
        with tf.GradientTape() as tape:
            source_pred,source_h = model(source_x_subset)
            target_pred,target_h = model(target_x_subset)
            loss = loss_func(source_y_subset,source_pred,target_y_subset,target_pred,target_h,source_h,coral_lam)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_performance=(0,0,0,0) # keeping track of the best model epoch: (val_acc,val_bal-acc,test_acc,test_bal-acc)
    used_cutoff=False # bool: tracks whether the cutoff was used

    if model_type=="semi_supervised": # different loss functions depending on whether target-domain data is incorporated into task loss
        loss_func = coral_loss_semi_supervised
    elif model_type=="unsupervised":
        loss_func = coral_loss_unsupervised

    for epoch_i in range(n_epochs):
        for i in range(0,len(source_x),batch_size):
            source_x_train_subset = source_x[i:i+batch_size]
            source_y_train_subset = source_y[i:i+batch_size]
            target_x_train_subset = target_x[i%len(target_x):i%len(target_x)+batch_size] 
            target_y_train_subset = target_y[i%len(target_y):i%len(target_y)+batch_size]
            batch_loss = train_coral(model,optimizer,source_x_train_subset,source_y_train_subset,target_x_train_subset,target_y_train_subset,loss_func)
        
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
