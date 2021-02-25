""" DANN model components.

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


@tf.custom_gradient # defining the gradient-reversal layer
def reverse_grad(x):
    """ identity function during forward_pass: f(x)=x; -1 gradient: f'(x)=-1
    """
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y,custom_grad

class ReverseGrad(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def call(self,x):
        return reverse_grad(x)


def dann_model(input_dim):
    """ DANN model implementation
    """
    x = Input(shape=(input_dim))
    h = Dense(256,activation=None)(x)
    h = Activation('relu')(h)
    h_inner = ReverseGrad()(h) # reverses the gradient going to the class prediction
    class_pred = Dense(1,activation="sigmoid")(h_inner) # predicting the class(domain) of the input
    task_pred = Dense(1,activation="sigmoid")(h) # predicting sentiment label
    
    model = Model(inputs=[x],outputs=[task_pred,class_pred])
    return model


def dann_loss_semi_supervised(source_y_train_subset,source_task_pred,target_y_train_subset,target_task_pred,source_class,source_class_pred,target_class,target_class_pred,task_lam):
    """ DANN loss function
    args:
        task_lam: loss contribution for the task classification loss [0-1]
    """
    # task-specific loss:
    source_task_loss = BinaryCrossentropy()(source_y_train_subset,source_task_pred)
    target_task_loss = BinaryCrossentropy()(target_y_train_subset,target_task_pred)
    task_loss = 1/2*(target_task_loss+source_task_loss) # semi-supervised
    # class loss:
    target_class_loss = BinaryCrossentropy()(target_class,target_class_pred) # automatic avg over batch
    source_class_loss = BinaryCrossentropy()(source_class,source_class_pred)
    class_loss = 1/2*(target_class_loss+source_class_loss)
    
    total_loss = (task_lam*task_loss)+((1-task_lam)*class_loss)
    return total_loss


def dann_loss_unsupervised(source_y_train_subset,source_task_pred,target_y_train_subset,target_task_pred,source_class,source_class_pred,target_class,target_class_pred,task_lam):
    """ DANN loss function
    args:
        task_lam: loss contribution for the task classification loss [0-1]
    """
    # task-specific loss:
    source_task_loss = BinaryCrossentropy()(source_y_train_subset,source_task_pred)
    task_loss = source_task_loss # unsupervised
    # class loss:
    target_class_loss = BinaryCrossentropy()(target_class,target_class_pred) # automatic avg over batch
    source_class_loss = BinaryCrossentropy()(source_class,source_class_pred)
    class_loss = 1/2*(target_class_loss+source_class_loss)
    
    total_loss = (task_lam*task_loss)+((1-task_lam)*class_loss)
    return total_loss


def get_single_dann_model_simulation(model,source_x,source_y,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size,model_type,optimizer=Adam(lr=0.003),use_cutoff=False,cutoff=0.9):
    """ Training a single instance of the DANN model
    """
    @tf.function # this is required to be a nested function
    def train_dann(model,optimizer,source_x_train_subset,source_y_train_subset,target_x_train_subset,target_y_train_subset,source_class,target_class,loss_func,task_lam=0.75):
        """ Used to train the DANN model
        """
        with tf.GradientTape() as tape:
            source_task_pred,source_class_pred = model(source_x_train_subset)
            target_task_pred,target_class_pred = model(target_x_train_subset)
            loss = loss_func(source_y_train_subset,source_task_pred,target_y_train_subset,target_task_pred,source_class,source_class_pred,target_class,target_class_pred,task_lam)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_performance=(0,0,0,0) # keeping track of the best model epoch: (val_acc,val_bal-acc,test_acc,test_bal-acc)
    used_cutoff=False # bool: tracks whether the cutoff was used

    source_class = np.zeros((batch_size)).astype("float32") # labels for the class loss contribution
    target_class = np.ones((batch_size)).astype("float32")

    if model_type=="semi_supervised": # different loss functions depending on whether target-domain data is incorporated into task loss
        loss_func = dann_loss_semi_supervised
    elif model_type=="unsupervised":
        loss_func = dann_loss_unsupervised

    for epoch_i in range(n_epochs):
        for i in range(0,len(source_x),batch_size):
            source_x_train_subset = source_x[i:i+batch_size]
            source_y_train_subset = source_y[i:i+batch_size]
            target_x_train_subset = target_x[i%len(target_x):i%len(target_x)+batch_size]
            target_y_train_subset = target_y[i%len(target_y):i%len(target_y)+batch_size]
            batch_loss = train_dann(model,optimizer,source_x_train_subset,source_y_train_subset,target_x_train_subset,target_y_train_subset,source_class,target_class,loss_func)
        
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
