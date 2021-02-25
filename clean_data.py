""" Gathers and saves the partitions of data for all domains. This includes creating the initial subset of data for faster processing.

Author: Nathaniel Andre
"""

import os
import math
import torch
import time
import numpy as np
from sklearn.utils import shuffle
from transformers import DistilBertTokenizer,DistilBertModel


def main():
    clean_all_data()
    pass


def clean_all_data(data_dir="../data/"):
    """ saves the source,target train,val,test partitions for all domains
        -data is collected from partitioning the data from each domain into two chunks, separated at the midpoint
    """
    all_review_dirs = ["amazon_toys_reviews","amazon_electronics_reviews","amazon_kitchen_reviews","amazon_reviews","imdb_reviews","yelp_reviews"]

    for review_dir in all_review_dirs:
        for partition_n in [1,2]:
            s = time.time()
            print("--",review_dir,partition_n)
            get_data_splits(review_dir,partition_n,data_dir)
            print(round((time.time()-s)/60,2))


def get_model_embedding(inputs,bert_model,cutoff_len=510,cls_token=101,sep_token=102):
    """ returns CLS embedding given the input tokens
        -handles case when input size is larger than the specified input size for the model
    args:
        cutoff_len: approx. the maximum size of BERT model
        cls_token,sep_token: indices of tokens for cls,sep
    """
    inputs_len = inputs['input_ids'].shape[-1]-2
    
    if inputs_len <= cutoff_len:
        model_output = bert_model(**inputs).last_hidden_state.detach().numpy()[0,0,:] # getting CLS embedding
    else:
        old_input_ids = inputs['input_ids'].tolist()[0][1:-1] # without the cls,sep tokens
        old_att_mask = inputs['attention_mask'].tolist()[0][1:-1]
        num_segments = math.ceil(inputs_len/cutoff_len)
        segment_len = math.ceil(inputs_len/num_segments)
        
        model_inputs = [] # input segmented into approx.-equally sized chunks
        start_i = 0
        for i in range(1,num_segments+1):
            if i==num_segments:
                end_i = inputs_len
                model_inputs.append({'input_ids':torch.Tensor([[cls_token]+old_input_ids[start_i:end_i]+[sep_token]]).to(torch.int64),'attention_mask':torch.Tensor([[1]+old_att_mask[start_i:end_i]+[1]]).to(torch.int64)})
            else:
                end_i = start_i+segment_len
                model_inputs.append({'input_ids':torch.Tensor([[cls_token]+old_input_ids[start_i:end_i]+[sep_token]]).to(torch.int64),'attention_mask':torch.Tensor([[1]+old_att_mask[start_i:end_i]+[1]]).to(torch.int64)})
                start_i = end_i
        
        model_outputs = []
        for inputs in model_inputs:
            model_outputs.append(bert_model(**inputs).last_hidden_state.detach().numpy()[0,0,:])
        model_output = np.average(np.stack(model_outputs),axis=0) # average CLS embedding over segments
    
    return model_output


def get_embeddings(lines,tokenizer,bert_model):
    """ converts the review strings into list of embeddings using the BERT encoder
        -concatenates the sentiment label to the end of the list including the embedding
    """
    embeddings = []

    for line in lines:
        label = int(line[0])
        review = line[2:].strip()
        review = review.lower() # only pre-processing component
        inputs = tokenizer(review,return_tensors="pt")
        embedding = get_model_embedding(inputs,bert_model)
        embeddings.append((embedding,[label]))

    return embeddings


def save_source_split(pos_reviews,neg_reviews,review_dir,partition_n,data_dir,source_train_size):
    """ saves to memory the train split for the source data
    """
    source_train = pos_reviews[:source_train_size//2] + neg_reviews[:source_train_size//2]
    source_train = np.vstack(source_train).astype("float32")
    source_train = shuffle(source_train,random_state=10) # fixed state
    np.save(data_dir+review_dir+"/source_train_p{}.npy".format(partition_n),source_train)


def save_target_split(pos_reviews,neg_reviews,review_dir,partition_n,data_dir,target_train_sizes,target_val_size,target_test_size):
    """ saves to memory the train,val,test splits for the target data
    """
    data_i = 0 # keeps track of index in pos,neg reviews lists to ensure no data points are used more than once

    for train_size in target_train_sizes:
        target_train = pos_reviews[data_i:data_i+train_size//2] + neg_reviews[data_i:data_i+train_size//2]
        target_train = np.vstack(target_train).astype("float32")
        target_train = shuffle(target_train,random_state=10) # fixed state
        np.save(data_dir+review_dir+"/target_train_n{}_p{}.npy".format(train_size,partition_n),target_train)
        data_i += train_size//2

    target_val = pos_reviews[data_i:data_i+target_val_size//2] + neg_reviews[data_i:data_i+target_val_size//2]
    target_val = np.vstack(target_val).astype("float32")
    target_val = shuffle(target_val,random_state=10)
    np.save(data_dir+review_dir+"/target_val_p{}.npy".format(partition_n),target_val)
    data_i += target_val_size//2

    target_test = pos_reviews[data_i:data_i+target_test_size//2] + neg_reviews[data_i:data_i+target_test_size//2]
    target_test = np.vstack(target_test).astype("float32")
    target_test = shuffle(target_test,random_state=10)
    np.save(data_dir+review_dir+"/target_test_p{}.npy".format(partition_n),target_test)
    data_i += target_test_size//2


def get_data_splits(review_dir,partition_n,data_dir,source_train_size=20000,target_train_sizes=[200,900],target_val_size=100,target_test_size=5000):
    """ gets the source(train) and target(train,val,test) splits for a given domain and partition num.
    args:
        source_train_size: num. examples to include in the source-domain training set
        target_train_sizes: num. examples to include in target-domain training sets
        target_val_size: num. examples to include in target-domain val/dev set (same val set for all training sets, for given partition_n)
        target_test_size: num. examples to include in target-domain test set (same test set for all training sets, for given partition_n)
    """
    with open(data_dir+review_dir+"/reviews.txt") as in_file:
        lines = in_file.readlines()

    if partition_n == 1: # partitioning the data into halves
        lines = lines[:len(lines)//2]
    else:
        lines = lines[len(lines)//2:]

    tokenizer = DistilBertTokenizer.from_pretrained(data_dir+"distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained(data_dir+"distilbert-base-uncased")
    embeddings = get_embeddings(lines,tokenizer,bert_model) # [(embedding,[label]),...]

    pos_reviews = [] # creating sets of positive and negative keywords
    neg_reviews = []
    for embedding_list in embeddings:
        if embedding_list[-1][0] == 0:
            neg_reviews.append(np.concatenate(embedding_list)) # (embedding,[label])
        elif embedding_list[-1][0] == 1:
            pos_reviews.append(np.concatenate(embedding_list))
        else:
            print("error")

    save_source_split(pos_reviews,neg_reviews,review_dir,partition_n,data_dir,source_train_size)
    save_target_split(pos_reviews,neg_reviews,review_dir,partition_n,data_dir,target_train_sizes,target_val_size,target_test_size)


def get_intra_data_subsets(cutoff=100000,data_dir="../data/"):
    """ generates a subset of each Amazon review dataset and saves to memory
    """
    amazon_review_info = [("amazon_toys_reviews","Toys_and_Games_5"),("amazon_electronics_reviews","Electronics_5"),("amazon_kitchen_reviews","Home_and_Kitchen_5")]

    for item in amazon_review_info:
        amazon_review_folder,amazon_review_fname = item

        n_valid=0 # counts the number of valid lines of data
        n_pos=0 # ensure half of the datapoints are of each sentiment
        n_neg=0
        data_to_save = [] # tuples:(score (0-1), review)
        data_file = open("{}{}/{}.json".format(data_dir,amazon_review_folder,amazon_review_fname))
        lines = data_file.readlines()

        for line in lines:
            line_json = json.loads(line)
            if 'overall' not in line_json or 'reviewText' not in line_json: # must have both a review and score
                continue
            rating,review = line_json['overall'],line_json['reviewText']
            if rating == 3:
                continue
            else:
                if rating <= 2 and n_neg < cutoff//2: # negative
                    data_to_save.append((0,review))
                    n_neg+=1
                    n_valid+=1
                elif rating >= 4 and n_pos < cutoff//2: # positive
                    data_to_save.append((1,review))
                    n_pos+=1
                    n_valid+=1
            if n_valid==cutoff:
                break

        data_to_save = shuffle(data_to_save,random_state=10) # fixed state
        data_file.close()
        with open("{}{}/reviews.txt".format(data_dir,amazon_review_folder),"w+") as out_file: # saving the data
            for dp in data_to_save:
                rating,review = dp
                out_file.write(str(rating)+" "+review.replace("\n"," ")+"\n")


def get_amazon_data_subset(data_dir="../data/"):
    """ generates the general Amazon subset which is an equal mix of each product type
    """
    all_data_paths = ["amazon_toys_reviews/","amazon_electronics_reviews/","amazon_kitchen_reviews/"]
    general_amazon_dataset = []

    for d_path in all_data_paths:
        with open(data_dir+d_path+"reviews.txt") as data_file:
            lines = data_file.readlines()
            general_amazon_dataset += lines[:30000]
            
    general_amazon_dataset = shuffle(general_amazon_dataset,random_state=10) # fixed state
    with open(data_dir+"amazon_reviews/reviews.txt","w+") as out_file:
        for line in general_amazon_dataset:
            out_file.write(line)


def get_yelp_data_subset(cutoff=100000,data_dir="../data/"):
    """ generates the general Yelp subset
    """
    n_valid=0 # counts the number of valid lines of data
    n_pos=0 # ensure half of the datapoints are of each sentiment
    n_neg=0
    data_to_save = [] # tuples:(score (0-1), review)
    data_file = open(data_dir+"yelp_reviews/yelp_academic_dataset_review.json")
    lines = data_file.readlines()

    for line in lines:
        line_json = json.loads(line)
        if 'text' not in line_json or 'stars' not in line_json: # must have both a review and score
            continue
        rating,review = line_json['stars'],line_json['text']
        if rating == 3:
            continue
        else:
            if rating <= 2 and n_neg < cutoff//2: # negative
                data_to_save.append((0,review))
                n_neg+=1
                n_valid+=1
            elif rating >= 4 and n_pos < cutoff//2: # positive
                data_to_save.append((1,review))
                n_pos+=1
                n_valid+=1
        if n_valid==cutoff:
            break
            
    data_to_save = shuffle(data_to_save,random_state=10) # fixed state
    data_file.close()
    with open(data_dir+"yelp_reviews/reviews.txt","w+") as out_file: # saving the data
        for dp in data_to_save:
            rating,review = dp
            out_file.write(str(rating)+" "+review.replace("\n"," ").replace("\r"," ")+"\n")


def get_imdb_data_subset(data_dir="../data/"):
    """ generates the IMDB subset
    """
    data_dir += "imdb_reviews/"
    data_to_save = []

    neg_fnames = [f for f in os.listdir(data_dir+"neg/") if f[-3:]=="txt"]
    for neg_fname in neg_fnames:
        with open(data_dir+"neg/"+neg_fname) as in_file:
            lines = in_file.readlines()
            data_to_save.append((0,lines[0]))
            
    pos_fnames = [f for f in os.listdir(data_dir+"pos/") if f[-3:]=="txt"]
    for pos_fname in pos_fnames:
        with open(data_dir+"pos/"+pos_fname) as in_file:
            lines = in_file.readlines()
            data_to_save.append((1,lines[0]))

    data_to_save = shuffle(data_to_save,random_state=10)
    with open(data_dir+"reviews.txt","w+") as out_file: # saving the data
        for dp in data_to_save:
            rating,review = dp
            review = review.replace("<br />"," ")
            out_file.write(str(rating)+" "+review.replace("\n"," ").replace("\r"," ")+"\n")


if __name__=="__main__":
    main()
