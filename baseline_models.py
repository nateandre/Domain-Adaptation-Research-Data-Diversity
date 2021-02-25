""" Implementations for simulating the target-only and baseline domain adaptation models (semi-supervised and unsupervised).

Author: Nathaniel Andre
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from dann import dann_model,get_single_dann_model_simulation
from mmd import mmd_model,get_single_mmd_model_simulation
from coral import coral_model,get_single_coral_model_simulation
import time
from modeling_utils import load_data,get_mean
import warnings
warnings.filterwarnings("ignore")


def compile_target_model(input_dim,optimizer):
    """ standard target-domain model architecture
        -fixed MLP architecture w/ a single hidden layer with 256 nodes
    """
    x = Input(shape=(input_dim))
    h = Dense(256,activation=None)(x)
    h_2 = Activation('relu')(h)
    out = Dense(1,activation='sigmoid')(h_2)
    
    model = Model(inputs=x,outputs=out)
    model.compile(loss='binary_crossentropy',optimizer=optimizer)
    return model


def get_single_target_model_simulation(model,x,y,x_val,y_val,x_test,y_test,n_epochs,batch_size,use_cutoff=False,cutoff=0.9):
    """ Training a single instance of the target domain model - also used to train the stacked model
    args:
        use_cutoff(bool): whether to cutoff training based on cutoff value
        cutoff: value for determining whether the model has finished training (based on val accuracy)
    """
    best_performance=(0,0,0,0) # keeping track of the best model epoch: (val_acc,val_bal-acc,test_acc,test_bal-acc)
    used_cutoff=False # bool: tracks whether the cutoff was used

    for epoch_i in range(n_epochs):
        for i in range(0,len(x),batch_size):
            x_train_subset = x[i:i+batch_size]
            y_train_subset = y[i:i+batch_size]
            batch_loss = model.train_on_batch(x_train_subset,y_train_subset)
        
        # getting the predictions for the val/test sets        
        val_pred = model(x_val).numpy()
        val_pred[val_pred >= 0.5]=1 ; val_pred[val_pred < 0.5]=0
        test_pred = model(x_test).numpy()
        test_pred[test_pred >= 0.5]=1 ; test_pred[test_pred < 0.5]=0
        
        # getting the acc/bal_acc metrics:  
        val_acc,val_bal_acc = round(accuracy_score(y_val,val_pred),4),round(balanced_accuracy_score(y_val,val_pred),4)    
        test_acc,test_bal_acc = round(accuracy_score(y_test,test_pred),4),round(balanced_accuracy_score(y_test,test_pred),4)

        if val_acc > best_performance[0]: # updating the best val performance for this run
            best_performance = (val_acc,val_bal_acc,test_acc,test_bal_acc)

        if use_cutoff:
            if val_acc>=cutoff:
                used_cutoff=True
                break

    return model,best_performance,used_cutoff


def get_target_model_simulation(all_data,target_data_name,n_simulations=5,n_epochs=20,batch_size=50,output_dir="../model_output/",save_runs=False):
    """ Simulation based on training a model with only target data
    args:
        all_data: list which contains the target train,val,test sets
        target_data_name: name of training data (e.g. amazon_toys)
        n_simulations: number of times to run the model from scratch (average results over)
        n_epochs: number of epochs to train the model for each instantiation
        output_dir: directory in which to store the model runs metrics
        save_runs(bool): whether to save data on the runs to memory, also controls printing info to console
    """

    x,y,x_val,y_val,x_test,y_test,_ = all_data

    if save_runs:
        out_file = open(output_dir+"target_only/"+target_data_name+".txt","w+") # file for outputting model runs
        print(target_data_name+"; target-only")

    all_model_runs = [] # stores the best performance metrics for each run for later averaging
    for _ in range(n_simulations):
        start = time.time()
        model = compile_target_model(input_dim=x.shape[-1],optimizer=Adam(lr=0.003))
        _,best_performance,_ = get_single_target_model_simulation(model,x,y,x_val,y_val,x_test,y_test,n_epochs,batch_size,use_cutoff=False)
        all_model_runs.append(best_performance)
        val_acc,val_bal_acc,test_acc,test_bal_acc = best_performance
        model_run_text = "Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{}\n".format(val_acc,val_bal_acc,test_acc,test_bal_acc)
        took = str(round((time.time()-start)/60,2))
        if save_runs:
            out_file.write(model_run_text)
            print("-"+took,end='',flush=True)

    # getting the average performance
    mean_val_acc = round(get_mean([dp[0] for dp in all_model_runs]),4)
    mean_val_bal_acc = round(get_mean([dp[1] for dp in all_model_runs]),4)
    mean_test_acc = round(get_mean([dp[2] for dp in all_model_runs]),4)
    mean_test_bal_acc = round(get_mean([dp[3] for dp in all_model_runs]),4)
    model_final_text = "AVERAGE: Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{}\n".format(mean_val_acc,mean_val_bal_acc,mean_test_acc,mean_test_bal_acc)
    if save_runs:
        out_file.write(model_final_text)
        out_file.close()
        print()

    return mean_val_acc,mean_test_acc


def run_standard_target_model_simulation(partition_n,target_train_size,target_data_name,output_dir,data_dir="../data/",save_runs=True):
    """ Wrapper which runs the simulation for the standard target-domain model
    """
    all_data = load_data(target_dir=target_data_name+"_reviews",source_dirs=[],partition_n=partition_n,target_train_size=target_train_size,data_dir=data_dir)

    mean_val_acc,mean_test_acc = get_target_model_simulation(all_data,target_data_name,output_dir=output_dir,save_runs=save_runs)
    return mean_val_acc,mean_test_acc


def get_domain_adaptation_model_simulation(all_data,target_data_name,source_data_name,model_name,n_simulations=5,n_epochs=10,batch_size=50,model_type="semi_supervised",output_dir="../model_output/",save_runs=False):
    """ Simulation of the standard domain adaptation model.
    args:
        all_data: list which contains the target train,val,test sets and the source train set
        target_data_name: name of target domain data (e.g. amazon_toys)
        source_data_name: name of source domain data (e.g. amazon_toys)
        model_name: name of the domain adaptation model (dann,mmd,coral)
        model_type: (e.g. unsupervised or semi_supervised)
        save_runs(bool): whether to save data on the runs to memory, also controls printing info to console
    """
    target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,source_data = all_data
    source_x,source_y = source_data

    source_to_target_text = source_data_name+" -> "+target_data_name+"; "+model_name
    if save_runs:
        out_file = open(output_dir+model_type+"/"+source_to_target_text+".txt","w+") # file for outputting model runs
        print(source_to_target_text+"; "+model_type)

    all_model_runs = [] # stores the best performance metrics for each run for later averaging
    for _ in range(n_simulations):
        start = time.time()
        if model_name=="dann":
            model = dann_model(input_dim=target_x.shape[-1])
            _,best_performance,_ = get_single_dann_model_simulation(model,source_x,source_y,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size,model_type,optimizer=Adam(lr=0.003),use_cutoff=False)
        elif model_name=="mmd":
            model = mmd_model(input_dim=target_x.shape[-1])
            _,best_performance,_ = get_single_mmd_model_simulation(model,source_x,source_y,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size,model_type,optimizer=Adam(lr=0.003),use_cutoff=False)
        elif model_name=="coral":
            model = coral_model(input_dim=target_x.shape[-1])
            _,best_performance,_ = get_single_coral_model_simulation(model,source_x,source_y,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size,model_type,optimizer=Adam(lr=0.003),use_cutoff=False)

        all_model_runs.append(best_performance)
        val_acc,val_bal_acc,test_acc,test_bal_acc = best_performance
        model_run_text = "Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{}\n".format(val_acc,val_bal_acc,test_acc,test_bal_acc)
        took = str(round((time.time()-start)/60,2))
        if save_runs:
            out_file.write(model_run_text)
            print("-"+took,end='',flush=True)

    # getting the average performance
    mean_val_acc = round(get_mean([dp[0] for dp in all_model_runs]),4)
    mean_val_bal_acc = round(get_mean([dp[1] for dp in all_model_runs]),4)
    mean_test_acc = round(get_mean([dp[2] for dp in all_model_runs]),4)
    mean_test_bal_acc = round(get_mean([dp[3] for dp in all_model_runs]),4)
    model_final_text = "AVERAGE: Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{}\n".format(mean_val_acc,mean_val_bal_acc,mean_test_acc,mean_test_bal_acc)
    if save_runs:
        out_file.write(model_final_text)
        out_file.close()
        print()
    
    return mean_val_acc,mean_test_acc


def run_standard_domain_adaptation_model_simulation(partition_n,target_train_size,target_data_name,source_data_name,model_name,model_type,output_dir,data_dir="../data/",save_runs=True):
    """ Wrapper which runs the simulations for the standard domain adaptation
    """
    all_data = load_data(target_dir=target_data_name+"_reviews",source_dirs=[source_data_name+"_reviews"],partition_n=partition_n,target_train_size=target_train_size,data_dir=data_dir)
    
    mean_val_acc,mean_test_acc = get_domain_adaptation_model_simulation(all_data,target_data_name,source_data_name,model_name=model_name,model_type=model_type,output_dir=output_dir,save_runs=save_runs)
    return mean_val_acc,mean_test_acc
