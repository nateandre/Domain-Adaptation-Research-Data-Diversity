""" Implementation for the multi-source stacked model.

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
from baseline_models import get_target_model_simulation,get_single_target_model_simulation,compile_target_model,get_domain_adaptation_model_simulation
import warnings
warnings.filterwarnings("ignore")


def get_single_da_model(source_x,source_y,source_num,source_cutoff,model_name,model_type,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size):
    """ returns the trained source-domain model
    args:
        source_num: 1 or 2, used for debugging
    """
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
            print("s"+source_num,end='',flush=True)
            prev_best_cutoffs.append(best_performance[0]) # val acc
            if len(prev_best_cutoffs)%10==0:
                new_source_cutoff=round(get_mean(prev_best_cutoffs),4)
                print("[{},{}]".format(source_cutoff,new_source_cutoff),end='',flush=True)
                source_cutoff=new_source_cutoff
                prev_best_cutoffs = []
    
    source_model = Model(inputs=model.layers[0].input,outputs=model.layers[1].output) # extracting feature extraction layers
    source_model.trainable=False
    return source_model,source_cutoff


def get_mean_corr_values(h1,h2):
    """ calculates correlation metric - the mean of maximum correlations per row
    """
    corr_matrix = tfp.stats.correlation(h1,h2)
    corr_matrix = np.abs(corr_matrix.numpy()) # neg. and pos. correlation treated as equivalent
    corr_matrix[np.isnan(corr_matrix)]=-np.inf # remove nan
    arg_max = np.argmax(corr_matrix,axis=1)
    max_corr_values_per_row = corr_matrix[np.array([i for i in range(256)]),arg_max]
    mean_max_corr_values = round(float(np.mean(max_corr_values_per_row[~np.isnan(max_corr_values_per_row)])),4)
    return mean_max_corr_values


def compile_multi_stacked_model(source_model_1,source_model_2,target_model,input_dim,optimizer,reg_lam):
    """ multi-source stacked model implementation
        -logistic regression
    """
    x = Input(shape=(input_dim))
    source_1_h = source_model_1(x) # DA model
    source_2_h = source_model_2(x)
    target_h = target_model(x)
    combined_h = Concatenate()([source_1_h,source_2_h,target_h])
    
    h = combined_h
    out = Dense(1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(reg_lam))(h) # using L2 regularization
    
    model = Model(inputs=x,outputs=out)
    model.compile(loss='binary_crossentropy',optimizer=optimizer)
    return model


def get_single_multi_stacked_model_simulation(source_model_1,source_model_2,target_model,n_stacked_simulations,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_stacked_epochs,batch_size,reg_lam):
    """ Runs an instance of a multi-stacked model given trained source DA and target models and reg_lam
    """
    this_simulation_runs = []
    for _ in range(n_stacked_simulations): # helps account for random initialization
        model = compile_multi_stacked_model(source_model_1,source_model_2,target_model,input_dim=target_x.shape[-1],optimizer=Adam(lr=0.003),reg_lam=reg_lam)
        _,best_performance,_ = get_single_target_model_simulation(model,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_stacked_epochs,batch_size,use_cutoff=False)
        this_simulation_runs.append(best_performance)

    run_val_acc = round(get_mean([dp[0] for dp in this_simulation_runs]),4)
    run_val_bal_acc = round(get_mean([dp[1] for dp in this_simulation_runs]),4)
    run_test_acc = round(get_mean([dp[2] for dp in this_simulation_runs]),4)
    run_test_bal_acc = round(get_mean([dp[3] for dp in this_simulation_runs]),4)
    return run_val_acc,run_val_bal_acc,run_test_acc,run_test_bal_acc


def get_multi_stacked_model_simulation(partition_n,target_train_size,target_data_name,target_cutoff_dict,source_data_name_1,source_data_name_2,source_cutoff_dict,model_name,n_simulations=5,n_epochs=10,batch_size=50,n_stacked_simulations=2,n_stacked_epochs=20,model_type="unsupervised",data_dir="../data/",output_dir="../model_output/",l2_lam_values=[0.0,0.01,0.1,1.0],source_cutoff=20000,use_source_cutoff=False):
    """ Simulation for the stacked model - for the case in which there are two source-domain datasets
    args:
        source_data_name_1,source_data_name_2: names of the two source domain datasets
        target_cutoff_dict: holds cutoffs for target-only model (key: [target_data_name])
        source_cutoff_dict: holds cutoffs for DA models (key: [source-domain-used]_[target-domain]_[model_name])
        l2_lam_values: the lam. values for l2 regularization
        n_epochs: number of epochs to train the source DA models
        n_stacked_epochs: number of epochs to train the stacked model and the target model
        source_cutoff: the number of source-domain datapoints to use from each domain
    """
    all_data = load_data(target_dir=target_data_name+"_reviews",source_dirs=[source_data_name_1+"_reviews",source_data_name_2+"_reviews"],partition_n=partition_n,target_train_size=target_train_size,data_dir=data_dir,source_cutoff=source_cutoff,use_source_cutoff=use_source_cutoff)
    target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,source_data = all_data
    source_x_1,source_y_1, source_x_2,source_y_2 = source_data

    source_to_target_text = source_data_name_1+";"+source_data_name_2+" -> "+target_data_name
    out_files = [open(output_dir+"multi_stacked_"+model_type+"/"+source_to_target_text+"; lam="+str(lam)+"; "+model_name+".txt","w+") for lam in l2_lam_values] # files for outputting model runs
    print(source_to_target_text+"; "+model_name+"; "+model_type+"; multi-stacked")

    # getting the cutoffs for all models:
    target_cutoff = target_cutoff_dict[target_data_name]
    source_cutoff_1 = source_cutoff_dict[source_data_name_1+"_"+target_data_name+"_"+model_name]
    source_cutoff_2 = source_cutoff_dict[source_data_name_2+"_"+target_data_name+"_"+model_name]

    all_model_runs = [[] for _ in range(len(l2_lam_values))] # stores the best performance metrics for each run for later averaging; includes corr. metrics
    for _ in range(n_simulations):
        start = time.time()
        print("-",end='',flush=True)

        # training two source-domain DA models:
        source_model_1,source_cutoff_1 = get_single_da_model(source_x_1,source_y_1,"1",source_cutoff_1,model_name,model_type,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size)
        source_model_2,source_cutoff_2 = get_single_da_model(source_x_2,source_y_2,"2",source_cutoff_2,model_name,model_type,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_epochs,batch_size)

        # secondly training the target-domain model:
        used_cutoff = False
        while not used_cutoff: # keep running the model until the training was properly cutoff
            model = compile_target_model(input_dim=target_x.shape[-1],optimizer=Adam(lr=0.003))
            model,_,used_cutoff = get_single_target_model_simulation(model,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_stacked_epochs,batch_size,use_cutoff=True,cutoff=target_cutoff)
            if not used_cutoff: # debugging
                print("t",end='',flush=True)

        target_model = Model(inputs=model.layers[0].input,outputs=model.layers[1].output) # extracting feature extraction layers
        target_model.trainable=False

        # getting correlation metrics between the feature representations of the sources and target models for the target test set:
        target_model_h = target_model(target_x_test).numpy()
        source_model_1_h = source_model_1(target_x_test).numpy()
        source_model_2_h = source_model_2(target_x_test).numpy()
        s1_t_corr = get_mean_corr_values(source_model_1_h,target_model_h)
        s2_t_corr = get_mean_corr_values(source_model_2_h,target_model_h)
        s1_s2_corr = get_mean_corr_values(source_model_1_h,source_model_2_h)

        # lastly training the stacked model with the various l2_lam_values:
        for i,reg_lam in enumerate(l2_lam_values):
            run_val_acc,run_val_bal_acc,run_test_acc,run_test_bal_acc = get_single_multi_stacked_model_simulation(source_model_1,source_model_2,target_model,n_stacked_simulations,target_x,target_y,target_x_val,target_y_val,target_x_test,target_y_test,n_stacked_epochs,batch_size,reg_lam)

            this_simulation_run_best_performance = (run_val_acc,run_val_bal_acc,run_test_acc,run_test_bal_acc,s1_t_corr,s2_t_corr,s1_s2_corr)      
            all_model_runs[i].append(this_simulation_run_best_performance)

            if reg_lam==0.0: # the corr. values remain the same regardless of the reg. lam.
                model_run_text = "Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{}; CORR mean: s1,t:{} s2,t:{} s1,s2:{}\n".format(run_val_acc,run_val_bal_acc,run_test_acc,run_test_bal_acc,s1_t_corr,s2_t_corr,s1_s2_corr)
            else:
                model_run_text = "Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{};\n".format(run_val_acc,run_val_bal_acc,run_test_acc,run_test_bal_acc)
            out_files[i].write(model_run_text)

        took = str(round((time.time()-start)/60,2))
        print(took,end='',flush=True)
    print()

    all_test_acc_file = open(output_dir+"multi_stacked_"+model_type+"/all_test_averages/"+source_to_target_text+"; "+model_name+".txt","w+")
    all_test_string = "AVERAGE: Target TEST acc" # store the average test acc for each lam. value
    final_mean_test_acc_per_lam = [] # to return
    for i,reg_lam in enumerate(l2_lam_values):
        mean_val_acc = round(get_mean([dp[0] for dp in all_model_runs[i]]),4)
        mean_val_bal_acc = round(get_mean([dp[1] for dp in all_model_runs[i]]),4)
        mean_test_acc = round(get_mean([dp[2] for dp in all_model_runs[i]]),4)
        mean_test_bal_acc = round(get_mean([dp[3] for dp in all_model_runs[i]]),4)
        all_test_string += ("; lam={},acc:{}".format(reg_lam,mean_test_acc))
        final_mean_test_acc_per_lam.append((reg_lam,mean_test_acc))

        if reg_lam == 0.0: # the corr. values remain the same regardless of the reg. lam.
            mean_s1_t_corr = round(get_mean([dp[4] for dp in all_model_runs[i] if not np.isnan(dp[4])]),4)
            mean_s2_t_corr = round(get_mean([dp[5] for dp in all_model_runs[i] if not np.isnan(dp[5])]),4)
            mean_s1_s2_corr = round(get_mean([dp[6] for dp in all_model_runs[i] if not np.isnan(dp[6])]),4)
            model_final_text = "AVERAGE: Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{}; CORR mean: s1,t:{} s2,t:{} s1,s2:{}\n".format(mean_val_acc,mean_val_bal_acc,mean_test_acc,mean_test_bal_acc,mean_s1_t_corr,mean_s2_t_corr,mean_s1_s2_corr)
            final_mean_s1_s2_corr = mean_s1_s2_corr
        else:
            model_final_text = "AVERAGE: Target; VAL: acc:{} bal_acc:{}; TEST: acc:{} bal_acc:{};\n".format(mean_val_acc,mean_val_bal_acc,mean_test_acc,mean_test_bal_acc)

        out_files[i].write(model_final_text)
        out_files[i].close()

    all_test_acc_file.write(all_test_string+'\n')
    all_test_acc_file.close()

    return final_mean_test_acc_per_lam,final_mean_s1_s2_corr
