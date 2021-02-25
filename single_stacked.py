""" Implementation for the single-source stacked model.

Author: Nathaniel Andre
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input,Dense,Activation,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from dann import dann_model,get_single_dann_model_simulation
from mmd import mmd_model,get_single_mmd_model_simulation
from coral import coral_model,get_single_coral_model_simulation
import time
from modeling_utils import load_data,get_mean
from baseline_models import get_target_model_simulation,get_single_target_model_simulation,compile_target_model
import warnings
warnings.filterwarnings("ignore")


def compile_stacked_model(source_model,target_model,input_dim,optimizer):
    """ stacked model implementation
        -logistic regression
    """
    x = Input(shape=(input_dim))
    source_h = source_model(x) # DA model
    target_h = target_model(x)
    combined_h = Concatenate()([source_h,target_h])
    
    h = combined_h
    out = Dense(1,activation='sigmoid')(h)
    
    model = Model(inputs=x,outputs=out)
    model.compile(loss='binary_crossentropy',optimizer=optimizer)
    return model


def get_stacked_model_simulation(partition_n,target_train_size,target_data_name,source_data_name,target_cutoff_dict,source_cutoff_dict,model_name,n_simulations=5,n_epochs=10,batch_size=50,n_stacked_simulations=2,n_stacked_epochs=20,model_type="semi_supervised",data_dir="../data/",output_dir="../model_output/"):
    """ Simulation for the stacked model - which uses the features extracted from both the target model and the DA model.
    args:
        target_cutoff_dict: holds val acc cutoff for target-only model (key: [target_data_name])(invariant to source domain and model type)
        source_cutoff_dict: holds the val acc cutoff for DA model training (key: [source_data_name]_[target_data_name]_[model_name])
        n_epochs: number of epochs to train the source DA model
        n_stacked_simulations: number of times to run final stacked model to average results for each trained source,target model pairing
        n_stacked_epochs: number of epochs to train the stacked model and the target model
    """
    all_data = load_data(target_dir=target_data_name+"_reviews",source_dirs=[source_data_name+"_reviews"],partition_n=partition_n,target_train_size=target_train_size,data_dir=data_dir)
    target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,source_data = all_data
    source_x,source_y = source_data

    source_to_target_text = source_data_name+" -> "+target_data_name+"; "+model_name
    out_file = open(output_dir+"stacked_"+model_type+"/"+source_to_target_text+".txt","w+") # file for outputting model runs
    print(source_to_target_text+"; "+model_type+"; stacked")

    source_cutoff = source_cutoff_dict[source_data_name+"_"+target_data_name+"_"+model_name] # used for DA model
    target_cutoff = target_cutoff_dict[target_data_name] # used for target-only model

    all_model_runs = [] # stores the best performance metrics for each run for later averaging; includes corr. metrics
    for _ in range(n_simulations):
        start = time.time()
        print("-",end='',flush=True)

        # first training the DA model:
        used_cutoff = False
        prev_best_cutoffs = []
        while not used_cutoff: # keep running the model until the training was properly cutoff
            if model_name=="dann":
                model = dann_model(input_dim=target_x.shape[-1])
                model,best_performance,used_cutoff = get_single_dann_model_simulation(model,source_x,source_y,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size,model_type,optimizer=Adam(lr=0.003),use_cutoff=True,cutoff=source_cutoff)
            elif model_name=="mmd":
                model = mmd_model(input_dim=target_x.shape[-1])
                model,best_performance,used_cutoff = get_single_mmd_model_simulation(model,source_x,source_y,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size,model_type,optimizer=Adam(lr=0.003),use_cutoff=True,cutoff=source_cutoff)
            elif model_name=="coral":
                model = coral_model(input_dim=target_x.shape[-1])
                model,best_performance,used_cutoff = get_single_coral_model_simulation(model,source_x,source_y,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size,model_type,optimizer=Adam(lr=0.003),use_cutoff=True,cutoff=source_cutoff)
            if not used_cutoff: # debugging
                print("s",end='',flush=True)
                prev_best_cutoffs.append(best_performance[0]) # val acc
                if len(prev_best_cutoffs)%10==0:
                    new_source_cutoff=round(get_mean(prev_best_cutoffs),4)
                    print("[{},{}]".format(source_cutoff,new_source_cutoff),end='',flush=True)
                    source_cutoff=new_source_cutoff
                    prev_best_cutoffs = []
        
        source_model = Model(inputs=model.layers[0].input,outputs=model.layers[1].output) # extracting feature extraction layers
        source_model.trainable=False

        # secondly training the target-domain model:
        used_cutoff = False
        while not used_cutoff: # keep running the model until the training was properly cutoff
            model = compile_target_model(input_dim=target_x.shape[-1],optimizer=Adam(lr=0.003))
            model,_,used_cutoff = get_single_target_model_simulation(model,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_stacked_epochs,batch_size,use_cutoff=True,cutoff=target_cutoff)
            if not used_cutoff: # debugging
                print("t",end='',flush=True)

        target_model = Model(inputs=model.layers[0].input,outputs=model.layers[1].output) # extracting feature extraction layers
        target_model.trainable=False

        # getting correlation metrics between the feature representations of the source and target models for the target test set:
        target_model_h = target_model(target_x_test).numpy()
        source_model_h = source_model(target_x_test).numpy()
        corr_matrix = tfp.stats.correlation(target_model_h,source_model_h)
        corr_matrix = np.abs(corr_matrix.numpy()) # neg. and pos. correlation treated as equivalent
        corr_matrix[np.isnan(corr_matrix)]=-np.inf # remove nan
        arg_max = np.argmax(corr_matrix,axis=1)
        max_corr_values_per_row = corr_matrix[np.array([i for i in range(256)]),arg_max]
        mean_max_corr_values = round(float(np.mean(max_corr_values_per_row[~np.isnan(max_corr_values_per_row)])),4)

        # lastly training the stacked model:
        this_simulation_runs = []
        for _ in range(n_stacked_simulations): # helps account for random initialization
            model = compile_stacked_model(source_model,target_model,input_dim=target_x.shape[-1],optimizer=Adam(lr=0.003))
            _,best_performance,_ = get_single_target_model_simulation(model,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_stacked_epochs,batch_size,use_cutoff=False)
            this_simulation_runs.append(best_performance)

        run_val_acc = round(get_mean([dp[0] for dp in this_simulation_runs]),4)
        run_val_bal_acc = round(get_mean([dp[1] for dp in this_simulation_runs]),4)
        run_test_acc = round(get_mean([dp[2] for dp in this_simulation_runs]),4)
        run_test_bal_acc = round(get_mean([dp[3] for dp in this_simulation_runs]),4)
        this_simulation_run_best_performance = (run_val_acc,run_val_bal_acc,run_test_acc,run_test_bal_acc,mean_max_corr_values)
        all_model_runs.append(this_simulation_run_best_performance)
        model_run_text = "Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{}; CORR: mean:{}\n".format(run_val_acc,run_val_bal_acc,run_test_acc,run_test_bal_acc,mean_max_corr_values)
        out_file.write(model_run_text)
        took = str(round((time.time()-start)/60,2))
        print(took,end='',flush=True)
    print()

    mean_val_acc = round(get_mean([dp[0] for dp in all_model_runs]),4)
    mean_val_bal_acc = round(get_mean([dp[1] for dp in all_model_runs]),4)
    mean_test_acc = round(get_mean([dp[2] for dp in all_model_runs]),4)
    mean_test_bal_acc = round(get_mean([dp[3] for dp in all_model_runs]),4)
    mean_mean_max_corr = round(get_mean([dp[4] for dp in all_model_runs if not np.isnan(dp[4])]),4)

    model_final_text = "AVERAGE: Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{}; CORR: mean:{}\n".format(mean_val_acc,mean_val_bal_acc,mean_test_acc,mean_test_bal_acc,mean_mean_max_corr)
    out_file.write(model_final_text)
    out_file.close()

    return mean_test_acc,mean_mean_max_corr
