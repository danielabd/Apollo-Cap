import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import SGD
from tqdm import tqdm
import operator
from zero_shot_style.model import mining
import wandb
import pickle
from datetime import datetime
from zero_shot_style.utils import parser, get_hparams
from zero_shot_style.create_dataset_from_twitter import clean_text

import os
from sklearn.model_selection import train_test_split
import timeit


def create_correct_df(df,num_of_labels,desired_labels):
    # labels_set_dict = {dmiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral}
    print("Creating corrected df...")
    start = timeit.default_timer()
    labels_set = df.columns[-num_of_labels:]
    #create new df
    list_of_labels = []
    fixed_list_of_texts = []
    uncleaned_list_of_labels = []
    uncleaned_list_of_texts = []
    for i in range(df.shape[0]):#go over all rows
        # if i==100: #todo:remove
        #     break
        if df.iloc[i, -num_of_labels-1]:# skip on example_very_unclear
            continue
        relevant_idxs_for_labels = np.where(df.iloc[i, -num_of_labels:].values == 1)
        if len(relevant_idxs_for_labels[0])>1: #skip on multi classes example
            continue
        labels = labels_set[relevant_idxs_for_labels[0]]
        for l in labels:
            if desired_labels==list:
                if l not in desired_labels:
                    continue
            try:
                # without clean_text
                uncleaned_list_of_texts.append(df['text'][i])
                uncleaned_list_of_labels.append(l)
            except:
                pass
            try:
                fixed_list_of_texts.append(clean_text(df['text'][i]))
                list_of_labels.append(l)
            except:
                pass
    fixed_df = pd.DataFrame({'label': list_of_labels, 'text': fixed_list_of_texts})
    uncleaned_df = pd.DataFrame({'label': uncleaned_list_of_labels, 'text': uncleaned_list_of_texts})

    stop = timeit.default_timer()
    print('Time to create correct df is: ', stop - start)
    return fixed_df,uncleaned_df

def create_csv_file_for_text(df_train,df_test,path_train,path_test):
    def pad_lists(text):
        max_len = 0
        for k in text:
            if len(text[k])>max_len:
                max_len = len(text[k])
        for k in text:
            if len(text[k]) < max_len:
                text[k].extend(['']*(max_len-len(text[k])))
        return text
    print('Starting to create visual csv files...')
    text = {}
    for i in range(df_train.shape[0]):
        if df_train.iloc[i, -2] not in text:
            text[df_train.iloc[i, -2]] = ['',df_train.iloc[i, -1]]
        else: #already there is a key of this label
            text[df_train.iloc[i, -2]].append(df_train.iloc[i, -1])
    #padding lists to the same size
    text = pad_lists(text)
    tmp_df = pd.DataFrame(text)
    print(f"saving visual csv file for training set in {path_train}")
    tmp_df.to_csv(path_train)
    text = {}
    for i in range(df_test.shape[0]):
        if df_test.iloc[i, -2] not in text:
            text[df_test.iloc[i, -2]] = ['',df_test.iloc[i, -1]]
        else: #already there is a key of this label
            text[df_test.iloc[i, -2]].append(df_test.iloc[i, -1])
    text = pad_lists(text)
    tmp_df = pd.DataFrame(text)
    print(f"saving visual csv file for test set in {path_test}")
    tmp_df.to_csv(path_test)
    print('finished.')

def main():
    print('Start!')
    args = parser.parse_args()
    config = get_hparams(args)
    if os.path.isfile(os.path.join(config['data_dir'],config['csv_file_name_train'])):
        df_train = pd.read_csv(os.path.join(config['data_dir'],config['csv_file_name_train']))
        df_test = pd.read_csv(os.path.join(config['data_dir'],config['csv_file_name_test']))
    else: #create df_train, df_test
        desired_labels = config['desired_labels']
        data_file = config['data_file']
        if type(data_file) == list:
            s_df = pd.read_csv(os.path.join(config['data_dir'], data_file[0]))
            for f in data_file[1:]:
                datapath = os.path.join(config['data_dir'], f)
                cur_df = pd.read_csv(datapath)
                s_df = pd.concat([s_df, cur_df], axis=0, ignore_index=True)
        else:
            datapath = os.path.join(config['data_dir'], data_file)
            s_df = pd.read_csv(datapath)

        print(s_df.head())
        #  df.groupby(['User']).size().plot.bar()
        if config['data_name'] == 'go_emotions':
            num_of_labels = 28
            df,uncleaned_df = create_correct_df(s_df, num_of_labels, desired_labels)
        elif config['data_name'] == 'Twitter': # change titles to Label and text
            s_df = s_df.rename(columns={'User': 'label', 'Tweet': 'text'})
            df = s_df

        print(f"Working on {config['data_name']} data. Splitting DB to train, val and test data frames.")
        df_train, df_test = train_test_split(df, test_size = 0.15, random_state = 42)
        # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),  # todo check sklearn split data func - keeps proportions between classes across all splits
        #                                      [int(.8 * len(df)), int(.9 * len(df))])
        # print(len(df_train), len(df_val), len(df_test))
        print(f'len of train = {len(df_train)},len of test = {len(df_test)}')
        print(f"saving data file splitted to train and test sets to: {os.path.join(config['data_dir'],config['csv_file_name_train'])}\n{os.path.join(config['data_dir'],config['csv_file_name_test'])}")
        df_train.to_csv(os.path.join(config['data_dir'],config['csv_file_name_train']))
        df_test.to_csv(os.path.join(config['data_dir'],config['csv_file_name_test']))

    path_train = os.path.join(config['data_dir'], config['visual_csv_file_name_train'])
    path_test = os.path.join(config['data_dir'], config['visual_csv_file_name_test'])
    # create_csv_file_for_text(df_train, df_test, path_train, path_test)

    #create for uncleaned text
    if config['data_name'] == 'go_emotions':
        desired_labels = config['desired_labels']
        data_file = config['data_file']
        if type(data_file) == list:
            s_df = pd.read_csv(os.path.join(config['data_dir'], data_file[0]))
            for f in data_file[1:]:
                datapath = os.path.join(config['data_dir'], f)
                cur_df = pd.read_csv(datapath)
                s_df = pd.concat([s_df, cur_df], axis=0, ignore_index=True)
        else:
            datapath = os.path.join(config['data_dir'], data_file)
            s_df = pd.read_csv(datapath)
        num_of_labels = 28
        df, uncleaned_df = create_correct_df(s_df, num_of_labels, desired_labels)
        uncleaned_df_train, uncleaned_df_test = train_test_split(df, test_size=0.15, random_state=42)
        path_train = os.path.join(config['data_dir'], 'uncleaned_'+config['visual_csv_file_name_train'])
        path_test = os.path.join(config['data_dir'], 'uncleaned_'+config['visual_csv_file_name_test'])
        create_csv_file_for_text(uncleaned_df_train, uncleaned_df_test, path_train, path_test)

    print('  finish!')


if __name__ == '__main__':
    main()