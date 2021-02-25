""" Used to run the simulations of the model performances and produce the output metrics.

Author: Nathaniel Andre
"""
import os
import json
from baseline_models import run_standard_target_model_simulation,run_standard_domain_adaptation_model_simulation
from single_stacked import get_stacked_model_simulation
from multi_stacked import get_multi_stacked_model_simulation


def main():
    """
    """
    target_train_size=200
    partition_n=2
    _=make_directories(partition_n,target_train_size)
    target_model_simulation(partition_n,target_train_size,output_dir="../model_output/n{}_p{}/".format(target_train_size,partition_n))
    standard_da_simulation(partition_n,target_train_size,output_dir="../model_output/n{}_p{}/".format(target_train_size,partition_n))
    single_stacked_simulation(partition_n,target_train_size,output_dir="../model_output/n{}_p{}/".format(target_train_size,partition_n))
    multi_stacked_simulation(partition_n,target_train_size,output_dir="../model_output/n{}_p{}/".format(target_train_size,partition_n))

    ##run_entire_simulation(partition_n,target_train_size)
    pass


def run_entire_simulation(partition_n,target_train_size,output_dir="../model_output/"):
    """ Runs entire simulation for a given partition and target training size
    """
    output_dir = make_directories(partition_n,target_train_size,output_dir)

    # calling the functions for each of the sub-simulations:
    target_model_simulation(partition_n,target_train_size,output_dir)
    print("--------------------------------------------------\n")
    standard_da_simulation(partition_n,target_train_size,output_dir)
    print("--------------------------------------------------\n")
    single_stacked_simulation(partition_n,target_train_size,output_dir)
    print("--------------------------------------------------\n")
    multi_stacked_simulation(partition_n,target_train_size,output_dir)


def make_directories(partition_n,target_train_size,output_dir="../model_output/"):
    """ Creates model output directories and returns path
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = output_dir+"n{}_p{}/".format(target_train_size,partition_n) # folder to store results for this simulation
    os.mkdir(output_dir)
    sub_folder_names = ['semi_supervised','stacked_semi_supervised','target_only','unsupervised','stacked_unsupervised','multi_stacked_unsupervised','cutoff_dicts','test_performance']
    for sub_folder_name in sub_folder_names:
        os.mkdir(output_dir+sub_folder_name)
    os.mkdir(output_dir+"multi_stacked_unsupervised/all_test_averages")
    return output_dir


def target_model_simulation(partition_n,target_train_size,output_dir):
    """ Runs the target-only simulation
    """
    #target_datasets = ["amazon_toys","amazon_electronics","amazon_kitchen"]
    target_datasets = ["amazon","imdb","yelp"]

    target_cutoff_dict = {}
    performance_csv = open(output_dir+"test_performance/target_only.csv","w+") # stores test accuracy
    performance_csv.write("data_description,test_accuracy\n")

    for target_data_name in target_datasets:
        mean_val_acc,mean_test_acc = run_standard_target_model_simulation(partition_n=partition_n,target_train_size=target_train_size,target_data_name=target_data_name,output_dir=output_dir)
        performance_csv.write("{},{}\n".format(target_data_name,mean_test_acc))
        target_cutoff_dict[target_data_name]=mean_val_acc

    performance_csv.close()
    with open(output_dir+'cutoff_dicts/target_cutoff_dict.json',"w+") as out_file:
        json.dump(target_cutoff_dict,out_file)


def standard_da_simulation(partition_n,target_train_size,output_dir):
    """ Runs the simulations for the standard domain adaptation techniques (unsupervised and semi-supervised)
    """
    #source_target_pairs = [("amazon_electronics","amazon_toys"),("amazon_kitchen","amazon_toys"),("amazon_toys","amazon_electronics"),("amazon_kitchen","amazon_electronics"),("amazon_toys","amazon_kitchen"),("amazon_electronics","amazon_kitchen")]
    source_target_pairs = [("yelp","amazon"),("imdb","amazon"),("amazon","imdb"),("yelp","imdb"),("amazon","yelp"),("imdb","yelp")]
    model_names = ["mmd","coral","dann"]
    
    performance_csv = open(output_dir+"test_performance/semi_supervised_da.csv","w+")
    performance_csv.write("data_description,test_accuracy\n")
    semi_cutoff_dict = {} # used for the single-source stacked model

    for model_name in model_names: # looping over DA model types
        performance_csv.write(model_name+"\n")
        for pair in source_target_pairs:
            source_data_name,target_data_name = pair
            mean_val_acc,mean_test_acc = run_standard_domain_adaptation_model_simulation(partition_n=partition_n,target_train_size=target_train_size,target_data_name=target_data_name,source_data_name=source_data_name,model_name=model_name,model_type="semi_supervised",output_dir=output_dir)
            performance_csv.write("{},{}\n".format(source_data_name+" -> "+target_data_name+"; "+model_name,mean_test_acc))
            semi_cutoff_dict[source_data_name+"_"+target_data_name+"_"+model_name]=mean_val_acc

    performance_csv.close()
    with open(output_dir+'cutoff_dicts/source_cutoff_dict_semisupervised.json',"w+") as out_file:
        json.dump(semi_cutoff_dict,out_file)
    
    performance_csv = open(output_dir+"test_performance/unsupervised_da.csv","w+")
    performance_csv.write("data_description,test_accuracy\n")
    un_cutoff_dict = {}

    for model_name in model_names: # looping over DA model types
        performance_csv.write(model_name+"\n")
        for pair in source_target_pairs:
            source_data_name,target_data_name = pair
            mean_val_acc,mean_test_acc = run_standard_domain_adaptation_model_simulation(partition_n=partition_n,target_train_size=target_train_size,target_data_name=target_data_name,source_data_name=source_data_name,model_name=model_name,model_type="unsupervised",output_dir=output_dir)
            performance_csv.write("{},{}\n".format(source_data_name+" -> "+target_data_name+"; "+model_name,mean_test_acc))
            un_cutoff_dict[source_data_name+"_"+target_data_name+"_"+model_name]=mean_val_acc

    performance_csv.close()
    with open(output_dir+'cutoff_dicts/source_cutoff_dict_unsupervised.json',"w+") as out_file:
        json.dump(un_cutoff_dict,out_file)


def single_stacked_simulation(partition_n,target_train_size,output_dir):
    """ Runs the simulation for the stacked model where there is a single source-domain
    """
    #source_target_pairs = [("amazon_electronics","amazon_toys"),("amazon_kitchen","amazon_toys"),("amazon_toys","amazon_electronics"),("amazon_kitchen","amazon_electronics"),("amazon_toys","amazon_kitchen"),("amazon_electronics","amazon_kitchen")]
    source_target_pairs = [("yelp","amazon"),("imdb","amazon"),("amazon","imdb"),("yelp","imdb"),("amazon","yelp"),("imdb","yelp")]
    model_names = ["mmd","coral","dann"]

    target_cutoff_dict = json.load(open(output_dir+'cutoff_dicts/target_cutoff_dict.json'))
    source_cutoff_dict_semisupervised = json.load(open(output_dir+"cutoff_dicts/source_cutoff_dict_semisupervised.json"))
    source_cutoff_dict_unsupervised = json.load(open(output_dir+"cutoff_dicts/source_cutoff_dict_unsupervised.json"))

    performance_csv = open(output_dir+"test_performance/semi_supervised_stacked_acc.csv","w+") # avg. test accuracy
    performance_csv.write("data_description,test_accuracy\n")
    corr_csv = open(output_dir+"test_performance/semi_supervised_stacked_corr.csv","w+") # avg. correlation metric btwn source & target features
    corr_csv.write("data_description,avg_correlation\n")

    for model_name in model_names: # looping over DA model types
        performance_csv.write(model_name+"\n")
        corr_csv.write(model_name+"\n")
        for pair in source_target_pairs:
            source_data_name,target_data_name = pair
            mean_test_acc,mean_mean_max_corr = get_stacked_model_simulation(partition_n=partition_n,target_train_size=target_train_size,target_data_name=target_data_name,source_data_name=source_data_name,model_name=model_name,target_cutoff_dict=target_cutoff_dict,source_cutoff_dict=source_cutoff_dict_semisupervised,model_type="semi_supervised",output_dir=output_dir)
            performance_csv.write("{},{}\n".format(source_data_name+" -> "+target_data_name+"; "+model_name,mean_test_acc))
            corr_csv.write("{},{}\n".format(source_data_name+" -> "+target_data_name+"; "+model_name,mean_mean_max_corr))

    performance_csv.close()
    corr_csv.close()

    performance_csv = open(output_dir+"test_performance/unsupervised_stacked_acc.csv","w+") # avg. test accuracy
    performance_csv.write("data_description,test_accuracy\n")
    corr_csv = open(output_dir+"test_performance/unsupervised_stacked_corr.csv","w+") # avg. correlation metric btwn source & target features
    corr_csv.write("data_description,avg_correlation\n")

    for model_name in model_names: # looping over DA model types
        performance_csv.write(model_name+"\n")
        corr_csv.write(model_name+"\n")
        for pair in source_target_pairs:
            source_data_name,target_data_name = pair
            mean_test_acc,mean_mean_max_corr = get_stacked_model_simulation(partition_n=partition_n,target_train_size=target_train_size,target_data_name=target_data_name,source_data_name=source_data_name,model_name=model_name,target_cutoff_dict=target_cutoff_dict,source_cutoff_dict=source_cutoff_dict_unsupervised,model_type="unsupervised",output_dir=output_dir)
            performance_csv.write("{},{}\n".format(source_data_name+" -> "+target_data_name+"; "+model_name,mean_test_acc))
            corr_csv.write("{},{}\n".format(source_data_name+" -> "+target_data_name+"; "+model_name,mean_mean_max_corr))

    performance_csv.close()
    corr_csv.close()


def multi_stacked_simulation(partition_n,target_train_size,output_dir):
    """ Runs the simulation for the multi-stacked model where there are multiple source-domains
        -only done with unsupervised DA
    """
    #multi_source_target_pairs = [("amazon_electronics","amazon_kitchen", "amazon_toys"),("amazon_toys","amazon_kitchen", "amazon_electronics"),("amazon_toys","amazon_electronics", "amazon_kitchen")]
    multi_source_target_pairs = [("imdb","yelp", "amazon"), ("amazon","yelp", "imdb"), ("amazon","imdb", "yelp")]
    model_names = ["mmd","coral","dann"]
    l2_lam_values=[0.0,0.01,0.05,0.1,1.0]

    performance_csv = open(output_dir+"test_performance/unsupervised_multi_stacked_acc.csv","w+") # avg. test accuracy
    performance_csv_text = "data_description"
    for lam in l2_lam_values:
        performance_csv_text+=",lam={}".format(lam)
    performance_csv.write(performance_csv_text+"\n")
    corr_csv = open(output_dir+"test_performance/unsupervised_multi_stacked_corr.csv","w+") # avg. correlation metric btwn source & target features
    corr_csv.write("data_description,avg_correlation\n")

    target_cutoff_dict = json.load(open(output_dir+'cutoff_dicts/target_cutoff_dict.json'))
    source_cutoff_dict = json.load(open(output_dir+"cutoff_dicts/source_cutoff_dict_unsupervised.json"))

    for model_name in model_names: # looping over DA model types
        performance_csv.write(model_name+"\n")
        corr_csv.write(model_name+"\n")
        for pair in multi_source_target_pairs:
            source_data_name_1,source_data_name_2,target_data_name = pair
            final_mean_test_acc_per_lam,mean_s1_s2_corr = get_multi_stacked_model_simulation(partition_n=partition_n,target_train_size=target_train_size,target_data_name=target_data_name,source_data_name_1=source_data_name_1,source_data_name_2=source_data_name_2,model_name=model_name,target_cutoff_dict=target_cutoff_dict,source_cutoff_dict=source_cutoff_dict,model_type="unsupervised",l2_lam_values=l2_lam_values,output_dir=output_dir)
            performance_csv_text = "{}+{} -> {}; {}".format(source_data_name_1,source_data_name_2,target_data_name,model_name)
            for lam,final_mean_test_acc in final_mean_test_acc_per_lam:
                performance_csv_text += ",{}".format(final_mean_test_acc)
            performance_csv.write(performance_csv_text+"\n")
            corr_csv.write("{}+{} -> {}; {},{}\n".format(source_data_name_1,source_data_name_2,target_data_name,model_name,mean_s1_s2_corr))

    performance_csv.close()
    corr_csv.close()


if __name__=="__main__":
    main()
