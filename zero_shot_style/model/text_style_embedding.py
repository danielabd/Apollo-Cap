import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertConfig
from torch import nn
from transformers import BertModel
from torch.optim import SGD
from tqdm import tqdm
import operator
from zero_shot_style.model import mining
import wandb
import pickle
from datetime import datetime
from zero_shot_style.create_dataset_from_twitter import clean_text
from itertools import combinations, product
from sklearn.metrics import roc_curve, auc
import os
import random
from sklearn.model_selection import train_test_split
import timeit
from torch.optim import Adam
# from numpy import linalg
# import clip
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

from zero_shot_style.utils import parser, get_hparams
# from zero_shot_style.utils import *

BERT_NUM_OF_LAYERS = 12
MAX_VAL_TRIPLET_LOSS = 100
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def get_args():
    parser.add_argument('--config_file', type=str, default=os.path.join('.', 'configs', 'text_style_embedding.yaml'),
                        help='full path to config file')
    parser.add_argument('--margin', type=float, default=0.4, help='description')
    parser.add_argument('--hidden_state_to_take', type=int, default=-2, help='hidden state of BERT totake')
    parser.add_argument('--last_layer_idx_to_freeze', type=int, default=-1, help='last_layer idx of BERT to freeze')
    parser.add_argument('--freeze_after_n_epochs', type=int, default=3, help='freeze BERT after_n_epochs')
    parser.add_argument('--scale_noise', type=float, default=0.04, help='scale of gaussian noise to add to the embedding vector of sentence')
    args = parser.parse_args()
    return args


class PosNegPairsDataset(torch.utils.data.Dataset):
    def __init__(self, pos_combinations_labels, neg_combinations_labels):
        self.pairs = torch.tensor(pos_combinations_labels + neg_combinations_labels)
        self.labels = torch.tensor([1] * len(pos_combinations_labels) + [0] * len(neg_combinations_labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.pairs[item], self.labels[item]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labels_set_dict, inner_batch_size=1, all_data=True):
        self.labels = [labels_set_dict[label] for label in df['category']]  # create list of idxs for labels
        self.labels_set = list(set(self.labels))
        self.texts = list(df['text'])  # df['Tweet'] #[text for text in df['Tweet']]
        self.batch_size_per_label = inner_batch_size
        self.all_data = all_data  # boolean
        pass

    def classes(self):
        return self.labels

    def __len__(self):
        if self.all_data:
            return len(self.labels)
        else:  # get samples from set of labels
            return len(self.labels_set)

    def __getitem__(self, item):
        if self.all_data:
            label = self.labels[item]
            text = self.texts[item]
            return text, label

        else:  # get samples from data
            label = self.labels_set[item]
            list_idxs_for_label = np.array(self.labels) == label
            full_tweets_list = list(operator.itemgetter(list_idxs_for_label)(np.array(self.texts)))
            batch_tweets = random.sample(full_tweets_list, min(len(full_tweets_list), self.batch_size_per_label))
            return batch_tweets, label

#
# # based on bert
# class TextStyleEmbed(nn.Module):
#     def __init__(self, dropout=0.05, device=torch.device('cpu'), hidden_state_to_take=-1, last_layer_idx_to_freeze=-1, scale_noise=0):
#         super(TextStyleEmbed, self).__init__()
#         bert_config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
#         self.bert = BertModel.from_pretrained('bert-base-cased', config=bert_config)
#         self.freeze_layers(last_layer_idx_to_freeze)
#         # for param in self.bert.parameters():
#         #   param.requires_grad = False
#         self.dropout = nn.Dropout(dropout)
#         # self.linear = nn.Linear(768, NUM_OF_CLASSES)
#         self.linear1 = nn.Linear(768, 128)
#         # self.linear2 = nn.Linear(128, NUM_OF_CLASSES)
#         self.relu = nn.ReLU()
#         self.hidden_state_to_take = hidden_state_to_take
#         self.scale_noise = scale_noise

    # def forward(self, input_id, mask):
    #     # _, x = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    #     outputs = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    #     hidden_states = outputs[2]
    #     # embedding_output = hidden_states[0]
    #     attention_hidden_states = hidden_states[1:]
    #     x = attention_hidden_states[self.hidden_state_to_take][:, 0, :]  # CLS output of self.hidden_state_to_take layer.
    #     x = self.dropout(x)
    #     x = self.linear1(x)
    #     # x = torch.nn.functional.normalize(x)
    #     # todo: remove comment
    #     # #add gaussian noise
    #     x = x + torch.randn_like(x) * self.scale_noise
    #     x = x / x.norm(dim=-1, keepdim=True)
    #     return x

# based on bert
class TextStyleEmbed(nn.Module):
    def __init__(self, dropout=0.05, device=torch.device('cpu'), hidden_state_to_take=-1,
                 last_layer_idx_to_freeze=BERT_NUM_OF_LAYERS, scale_noise=0):
        super(TextStyleEmbed, self).__init__()
        bert_config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-cased', config=bert_config)
        self.freeze_layers(last_layer_idx_to_freeze)
        self.hidden_state_to_take = hidden_state_to_take
        self.scale_noise = scale_noise
        # for param in self.bert.parameters():
        #   param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(768, NUM_OF_CLASSES)
        self.linear1 = nn.Linear(768, 128)
        # self.linear2 = nn.Linear(128, NUM_OF_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        # _, x = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        outputs = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        hidden_states = outputs[2]
        # embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]
        x = attention_hidden_states[self.hidden_state_to_take][:, 0, :]  # CLS output of self.hidden_state_to_take layer.
        x = self.relu(x) # todo: maybe I need to do it
        x = self.dropout(x)
        x = self.linear1(x)
        # x = torch.nn.functional.normalize(x)

        # add gaussian noise
        x = x + torch.randn_like(x) * self.scale_noise
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def freeze_layers(self, last_layer_idx_to_freeze=BERT_NUM_OF_LAYERS):
        '''

        #:param freeze_layers: list of layers num to freeze
        :param last_layer_idx_to_freeze: int
        :return:
        '''
        for layer_idx in range(BERT_NUM_OF_LAYERS):
            for param in list(self.bert.encoder.layer[layer_idx].parameters()):
                if layer_idx <= last_layer_idx_to_freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                # print(f"BERT layers up to {layer_idx} are frozen.")


# with torch.no_grad():
#     text_style_features = clip.tokenize(text).to(device)
# #based on clip
# class TextStyleEmbed(nn.Module):
#
#     def __init__(self, dropout=0.05,device=torch.device('cpu')):
#         super(TextStyleEmbed, self).__init__()
#         # Initialize CLIP
#         clip_checkpoints = './clip_checkpoints'
#         self.clip, clip_preprocess = clip.load("ViT-B/32", device=device,
#                                           download_root=clip_checkpoints, jit=False)  # clip_preprocess for images
#         #freeze clip model
#         for param in self.clip.parameters():
#             param.requires_grad = False
#         self.device = device
#         self.linear1 = nn.Linear(512, 128)
#         self.linear2 = nn.Linear(128, 128)
#         self.relu = nn.ReLU()
#
#     def forward(self, text):
#         # clip_texts = clip.tokenize(text, truncate=True).to(self.device)
#         # text_features = self.clip.encode_text(clip_texts)
#         inputs = clip.tokenize(text, truncate=True).to(self.device)
#         x = self.clip.encode_text(inputs)
#         x = self.linear1(x.float())
#         x = self.relu(x)
#         x = self.linear2(x)
#         text_features = x / x.norm(dim=-1, keepdim=True)
#         return text_features


def collate_fn(data):  # for model based on bert
    texts_list = []
    labels_list = []
    for list_for_label in data:
        if type(list_for_label[0]) == list:
            for text in list_for_label[0]:
                texts_list.append(text)
                labels_list.append(list_for_label[1])
        else:
            texts_list.append(list_for_label[0])
            labels_list.append(list_for_label[1])
    tokenized_texts_list = tokenizer(texts_list, padding='max_length', max_length=40, truncation=True,
                                     return_tensors="pt")
    return tokenized_texts_list, labels_list, texts_list


# def collate_fn(data): #for model based on clip
#     texts_list = []
#     labels_list = []
#     for list_for_label in data:
#         if type(list_for_label[0]) == list:
#             for text in list_for_label[0]:
#                 texts_list.append(text)
#                 labels_list.append(list_for_label[1])
#         else:
#             texts_list.append(list_for_label[0])
#             labels_list.append(list_for_label[1])
#     return labels_list, texts_list

def get_auc(all_embeddings, pos_combinations_labels, neg_combinations_labels):
    print("creating dataset of pos neg pairs...")
    pairs_data_set = PosNegPairsDataset(pos_combinations_labels, neg_combinations_labels)
    pairs_dataloader = torch.utils.data.DataLoader(pairs_data_set, batch_size=1000000, shuffle=False,
                                                   num_workers=0)  # config['num_workers'])
    total_distances = []
    total_labels = []
    print("calc distances for positive and negative pairs...")
    for step, (pairs, labels) in enumerate(
            tqdm(pairs_dataloader, desc="calc distances for positive and negative pairs", leave=False)):
        distances = calc_dist(all_embeddings, pairs)
        total_distances.extend(distances)
        total_labels.extend(labels)
    predictions = 1 - torch.tensor(total_distances) / 2
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(total_labels, predictions)
    roc_auc = auc(fpr, tpr)
    print(f'roc_auc = {roc_auc}\nfinished to evaluate.')
    return roc_auc


def plot_graph_on_all_data(df_data, total_outputs, total_labels_str, total_texts_list, title, tgt_file_vec_emb, save_vec_emb=False):
    print(f"Plotting graph for {title}...")
    if save_vec_emb:
        print("Calculate mean and median embedding vectors...")
        mean_embedding_vectors = {}
        median_embedding_vectors = {}
        std_avg_embedding_vectors = {}
        for label in set(df_data["category"]):
            vectors_embedding = total_outputs[np.where(np.array(total_labels_str) == label), :]
            median_vector_embedding = np.median(vectors_embedding[0], 0)
            median_embedding_vectors[label] = median_vector_embedding
            mean_vector_embedding = np.mean(vectors_embedding[0], 0)
            mean_embedding_vectors[label] = mean_vector_embedding
            std_avg_embedding_vectors[label] = np.mean(np.std(vectors_embedding[0], 0))

        print('Saving mean of embedding vectors to: ' + tgt_file_vec_emb['mean'] + '...')
        with open(tgt_file_vec_emb['mean'], 'wb') as fp:
            pickle.dump(mean_embedding_vectors, fp)
        print(f'Saving avg of std of embedding vectors to: ' + tgt_file_vec_emb['std'] + '...')
        with open(tgt_file_vec_emb['std'], 'wb') as fp:
            pickle.dump(std_avg_embedding_vectors, fp)
        # print(f'Saving median of embedding vectors to: '+tgt_file_vec_emb['median']+'...')
        # with open(tgt_file_vec_emb['median'], 'wb') as fp:
        #    pickle.dump(median_embedding_vectors, fp)
        print(f'Finished to save.')

    if os.path.exists(tgt_file_vec_emb['mean']):
        print("take mean and median vectors for plotting.")
        with open(tgt_file_vec_emb['mean'], "rb") as input_file:
            mean_embedding_vectors = pickle.load(input_file)
        # with open(tgt_file_vec_emb['median'], "rb") as input_file:
        #     median_embedding_vectors = pickle.load(input_file)

        total_outputs_with_representation = total_outputs
        for label in set(df_data["category"]):
            # insert mean and median to the beginning
            total_labels_str = [f'mean_{label}'] + total_labels_str
            total_texts_list = [f'mean_{label}'] + total_texts_list

            total_outputs_with_representation = np.concatenate(
                (np.expand_dims(np.array(mean_embedding_vectors[label]), axis=0),
                 total_outputs_with_representation), axis=0)

        total_outputs = total_outputs_with_representation

    labeldf = pd.DataFrame({'Label': total_labels_str})
    embdf = pd.DataFrame(total_outputs, columns=[f'emb{i}' for i in range(total_outputs.shape[1])])
    textdf = pd.DataFrame({'text': total_texts_list})
    all_data = pd.concat([labeldf, embdf, textdf], axis=1, ignore_index=True)
    all_data.columns = ['Label'] + [f'emb{i}' for i in range(total_outputs.shape[1])] + ['text']
    return all_data


def bu_plot_graph_on_all_data(df_data, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, batch_size,
                              title, tgt_file_vec_emb, all_data=False, save_vec_emb=False, num_workers=0,
                              desired_labels='all', pos_combinations_labels=None, neg_combinations_labels=None):
    print(f"Plotting graph for {title} for {desired_labels} labels...")
    if desired_labels == 'all':
        desired_df = df_data
    else:
        desired_df = pd.DataFrame()
        for l in desired_labels:
            l_df = df_data.iloc[np.where(np.array(df_data["category"]) == l)[0], :]
            desired_df = pd.concat([desired_df, l_df])
    data_set = Dataset(desired_df, labels_set_dict, inner_batch_size, all_data)
    eval_dataloader = torch.utils.data.DataLoader(data_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)
    model.to(device)
    model.eval()

    log_dict = {}
    total_outputs = []
    total_labels = []
    total_texts_list = []
    with torch.no_grad():
        for step, (tokenized_texts_list, labels, texts_list) in enumerate(
                tqdm(eval_dataloader, desc="evaluation", leave=False)):  # for model based on bert
            total_labels.extend(labels)
            outputs = model(tokenized_texts_list['input_ids'].to(device),
                            tokenized_texts_list['attention_mask'].to(device))  # model based on bert
            outputs = outputs.to(torch.device('cpu'))
            total_outputs.extend(outputs)
            total_texts_list.extend(texts_list)
        total_outputs = torch.stack(total_outputs)
        total_labels_str = [labels_idx_to_str[user_idx] for user_idx in total_labels]
    print(f"**************\nBefor mean, media, len(total_outputs)={len(total_outputs)},len(total_labels)={len(total_labels)},len(total_texts_list)={len(total_texts_list)}")
    if save_vec_emb:
        print("Calculate mean and median embedding vectors...")
        mean_embedding_vectors = {}
        median_embedding_vectors = {}
        for label in set(desired_df["category"]):
            vectors_embedding = total_outputs[np.where(np.array(total_labels_str) == label), :]
            # median_vector_embedding = np.median(vectors_embedding[0], 0)
            # median_embedding_vectors[label] = median_vector_embedding
            mean_vector_embedding = torch.mean(vectors_embedding[0], 0)
            mean_embedding_vectors[label] = mean_vector_embedding
        print('Saving mean of embedding vectors to: ' + tgt_file_vec_emb['mean'] + '...')
        with open(tgt_file_vec_emb['mean'], 'wb') as fp:
            pickle.dump(mean_embedding_vectors, fp)
        # print(f'Saving median of embedding vectors to: '+tgt_file_vec_emb['median']+'...')
        # with open(tgt_file_vec_emb['median'], 'wb') as fp:
        #    pickle.dump(median_embedding_vectors, fp)
        print(f'Finished to save.')

        print('print for wandb')
        # variables with mean
        total_outputs_with_representation = total_outputs.to(torch.device('cpu'))
        for label in set(desired_df["category"]):
            # #insert mean and median to the beginning
            # total_labels_str = [f'mean_{label}',f'median_{label}']+total_labels_str
            # total_texts_list = [f'mean_{label}', f'median_{label}'] + total_texts_list
            # total_outputs_with_representation = np.concatenate(
            #     (np.array([mean_embedding_vectors[label]]),
            #      np.array([median_embedding_vectors[label]]), total_outputs_with_representation), axis=0)
            # insert mean and median to the beginning
            total_labels_str = [f'mean_{label}'] + total_labels_str
            total_texts_list = [f'mean_{label}'] + total_texts_list
            # total_outputs_with_representation = np.concatenate(
            #     (np.array(mean_embedding_vectors[label]), total_outputs_with_representation), axis=0)
            total_outputs_with_representation = np.concatenate(
                (np.expand_dims(np.array(mean_embedding_vectors[label]), axis=0),
                 total_outputs_with_representation), axis=0)
        total_outputs = total_outputs_with_representation
    print(f"**************\nafter mean, median, len(total_outputs)={len(total_outputs)},len(total_labels)={len(total_labels)},len(total_texts_list)={len(total_texts_list)}")
    labeldf = pd.DataFrame({'Label': total_labels_str})
    embdf = pd.DataFrame(total_outputs, columns=[f'emb{i}' for i in range(total_outputs.shape[1])])
    textdf = pd.DataFrame({'text': total_texts_list})
    all_data = pd.concat([labeldf, embdf, textdf], axis=1, ignore_index=True)
    all_data.columns = ['Label'] + [f'emb{i}' for i in range(total_outputs.shape[1])] + ['text']
    log_dict[title] = all_data
    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print(f"cur time is: {cur_time}")
    print("send data to wb")
    wandb.log({title: all_data})  # todo log without commit
    if pos_combinations_labels:
        roc_auc = get_auc(total_outputs, pos_combinations_labels, neg_combinations_labels)
        return log_dict, roc_auc
    return log_dict

def plot_final_graph_after_training(device, config, path_for_best_model, labels_idx_to_str, tgt_file_vec_emb,df,dataloader, title, save_vec_emb=False):
    print(title+" dataset...")
    model = TextStyleEmbed(device=device, hidden_state_to_take = config['hidden_state_to_take'])
    if 'cuda' in device.type:
        checkpoint = torch.load(path_for_best_model, map_location='cuda:0')
    else:
        checkpoint = torch.load(path_for_best_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # for plot
    final_total_outputs = []
    final_total_labels = []
    final_total_texts_list = []
    for step, (tokenized_texts_list, labels, texts_list) in enumerate(
            pbar := tqdm(dataloader, desc=title, leave=False, )):  # model based on bert
        labels = torch.from_numpy(np.asarray(labels)).to(device)
        tokenized_texts_list = tokenized_texts_list.to(device)
        # outputs = model(tokenized_texts_list['input_ids'].to(device),
        #                 tokenized_texts_list['attention_mask'].to(device))  # model based on bert
        outputs = model(tokenized_texts_list['input_ids'],
                        tokenized_texts_list['attention_mask'])  # model based on bert
        # for plot
        final_total_labels.extend(labels.cpu().detach().numpy())
        final_total_outputs.extend(outputs.cpu().detach().numpy())
        final_total_texts_list.extend(texts_list)
    final_total_outputs = np.stack(final_total_outputs)
    final_total_labels_str = [labels_idx_to_str[i] for i in final_total_labels]
    return plot_graph_on_all_data(df, final_total_outputs, final_total_labels_str, final_total_texts_list,
                                                  "best model for " + title,
                                                  tgt_file_vec_emb, save_vec_emb)

def train(model, optimizer, df_train, df_val, labels_set_dict, labels_idx_to_str, path_for_saving_last_model,
          path_for_saving_best_model, device, tgt_file_vec_emb, config):
    best_val_all_triplet_loss_avg = MAX_VAL_TRIPLET_LOSS
    train_data_set, val_data_set = Dataset(df_train, labels_set_dict), Dataset(df_val, labels_set_dict)
    train_dataloader = torch.utils.data.DataLoader(train_data_set, collate_fn=collate_fn,
                                                   batch_size=config['batch_size'], shuffle=True,
                                                   num_workers=config['num_workers'])
    val_dataloader = torch.utils.data.DataLoader(val_data_set, collate_fn=collate_fn, batch_size=config['batch_size'],
                                                 shuffle=True, num_workers=config['num_workers'])

    print('Starting to train...')
    model = model.to(device)

    best_loss = 1e16
    log_dict = {}
    for epoch in range(config['epochs']):
        model.train()
        if epoch == config['freeze_after_n_epochs']:
            model.freeze_layers(BERT_NUM_OF_LAYERS)
        train_list_all_triplet_loss_batch = []
        train_list_positive_loss_batch = []
        train_list_fraction_positive_triplets_batch = []
        train_list_num_positive_triplets_batch = []
        # for plot
        train_total_outputs = []
        train_total_labels = []
        train_total_texts_list = []
        for step, (tokenized_texts_list, labels, texts_list) in enumerate(
                pbar := tqdm(train_dataloader, desc="Training", leave=False, )):  # model based on bert
            labels = torch.from_numpy(np.asarray(labels)).to(device)
            outputs = model(tokenized_texts_list['input_ids'].to(device),
                            tokenized_texts_list['attention_mask'].to(device))  # model based on bert

            # triplet loss
            loss, num_positive_triplets, num_valid_triplets, all_triplet_loss_avg = mining.online_mine_all(labels,
                                                                                                           outputs,
                                                                                                           float(config[
                                                                                                                     'margin']),
                                                                                                           device=device)
            fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
            # gradient step
            loss.backward()
            optimizer.step()
            # update locals
            train_list_all_triplet_loss_batch.append(all_triplet_loss_avg)
            train_list_positive_loss_batch.append(loss)
            train_list_fraction_positive_triplets_batch.append(fraction_positive_triplets)
            train_list_num_positive_triplets_batch.append(num_positive_triplets)
            # for plot
            # train_total_labels.extend(labels.cpu().data.numpy())
            train_total_labels.extend(labels.cpu().detach().numpy())
            train_total_outputs.extend(outputs.cpu().detach().numpy())
            # train_total_outputs.extend(outputs.to(torch.device('cpu')))
            train_total_texts_list.extend(texts_list)
        train_total_outputs = np.stack(train_total_outputs)
        train_total_labels_str = [labels_idx_to_str[i] for i in train_total_labels]

        train_epoch_avg_all_triplet_loss = np.mean(
            [loss_elem.item() for loss_elem in train_list_all_triplet_loss_batch])
        train_epoch_avg_positive_loss = np.mean([elem.item() for elem in train_list_positive_loss_batch])
        train_epoch_avg_fraction_positive_triplets = np.mean(
            [elem.item() for elem in train_list_fraction_positive_triplets_batch])
        train_epoch_avg_list_num_positive_triplets = np.mean(
            [elem.item() for elem in train_list_num_positive_triplets_batch])
        print(
            "\nEpoch, Training: {}/{} - Loss: {:.4f}".format(epoch, config['epochs'], train_epoch_avg_all_triplet_loss),
            '\n')
        log_dict = {'train/epoch': epoch,
                    'train/loss': train_epoch_avg_positive_loss,
                    'train/fraction_positive_triplets': train_epoch_avg_fraction_positive_triplets,
                    'train/num_positive_triplets': train_epoch_avg_list_num_positive_triplets,
                    'train/all_triplet_loss_avg': train_epoch_avg_all_triplet_loss}
        # save last model
        # print(f'Epoch = {epoch},Saving model to: {path_for_saving_last_model}...')
        # torch.save({"model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             }, path_for_saving_last_model)  # finally check on all data training

        # log_dict["train_plot"] = plot_graph_on_all_data(df_train, train_total_outputs, train_total_labels_str, train_total_texts_list, "train",
        #                                                 tgt_file_vec_emb, save_vec_emb=False)

        # evaluation
        model.eval()
        val_list_all_triplet_loss_batch = []
        val_list_positive_loss_batch = []
        val_list_fraction_positive_triplets_batch = []
        val_list_num_positive_triplets_batch = []
        # for plot
        val_total_outputs = []
        val_total_labels = []
        val_total_texts_list = []
        with torch.no_grad():
            for step, (tokenized_texts_list, labels, texts_list) in enumerate(
                    pbar := tqdm(val_dataloader, desc="Validation", leave=False, )):
                labels = torch.from_numpy(np.asarray(labels)).to(device)
                outputs = model(tokenized_texts_list['input_ids'].to(device),
                                tokenized_texts_list['attention_mask'].to(device))  # model based on bert

                # triplet loss
                loss, num_positive_triplets, num_valid_triplets, all_triplet_loss_avg = mining.online_mine_all(labels,
                                                                                                               outputs,
                                                                                                               float(
                                                                                                                   config[
                                                                                                                       'margin']),
                                                                                                               device=device)
                fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
                # update locals
                val_list_all_triplet_loss_batch.append(all_triplet_loss_avg)
                val_list_positive_loss_batch.append(loss)
                val_list_fraction_positive_triplets_batch.append(fraction_positive_triplets)
                val_list_num_positive_triplets_batch.append(num_positive_triplets)
                # for plot
                val_total_labels.extend(labels.cpu().detach().numpy())
                val_total_outputs.extend(outputs.cpu().detach().numpy())
                val_total_texts_list.extend(texts_list)

            val_total_outputs = np.stack(val_total_outputs)
            val_total_labels_str = [labels_idx_to_str[i] for i in val_total_labels]
            val_epoch_avg_all_triplet_loss = np.mean([loss_elem.item() for loss_elem in val_list_all_triplet_loss_batch])
            val_epoch_avg_positive_loss = np.mean([elem.item() for elem in val_list_positive_loss_batch])
            val_epoch_avg_fraction_positive_triplets = np.mean(
                [elem.item() for elem in val_list_fraction_positive_triplets_batch])
            val_epoch_avg_list_num_positive_triplets = np.mean(
                [elem.item() for elem in val_list_num_positive_triplets_batch])
            print(
                "\nEpoch, Validation: {}/{} - Loss: {:.4f}".format(epoch, config['epochs'], val_epoch_avg_all_triplet_loss),
                '\n')
            if val_epoch_avg_all_triplet_loss<best_val_all_triplet_loss_avg:
                best_val_all_triplet_loss_avg = val_epoch_avg_all_triplet_loss
            log_dict.update({'val/epoch': epoch,
                             'val/loss': val_epoch_avg_positive_loss,
                             'val/fraction_positive_triplets': val_epoch_avg_fraction_positive_triplets,
                             'val/num_positive_triplets': val_epoch_avg_list_num_positive_triplets,
                             'val/all_triplet_loss_avg': val_epoch_avg_all_triplet_loss,
                             'val/best_val_all_triplet_loss_avg': best_val_all_triplet_loss_avg})
            log_dict.update({"val_plot": plot_graph_on_all_data(df_val, val_total_outputs, val_total_labels_str, val_total_texts_list, "val",
                                                          tgt_file_vec_emb, save_vec_emb=False)})

        if val_epoch_avg_all_triplet_loss < best_loss:
            # save best model
            print(
                f'Val loss is improved. Epoch = {epoch}, cur best loss = {val_epoch_avg_all_triplet_loss} < {best_loss}. Saving the better model to: {path_for_saving_best_model}...')
            best_loss = val_epoch_avg_all_triplet_loss
            # save model
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        }, path_for_saving_best_model)  # finally check on all data training
            # log_dict["val_plot"] = plot_graph_on_all_data(df_val, val_total_outputs, val_total_labels_str, val_total_texts_list, "val",
            #                                               tgt_file_vec_emb, save_vec_emb=False)
        wandb.log(log_dict)

    # plot_train_data_with_best_model
    # plot_final_graph-after_training()
    print("Finished to train, plotting final embedding graphs of entire data set with vector embeddings")

    log_dict["final_train_plot"] = plot_final_graph_after_training(device, config, path_for_saving_best_model, labels_idx_to_str, tgt_file_vec_emb,df_train,train_dataloader, title="Train", save_vec_emb=True)
    log_dict["final_val_plot"] = plot_final_graph_after_training(device, config, path_for_saving_best_model, labels_idx_to_str, tgt_file_vec_emb,df_val,val_dataloader, title="Val", save_vec_emb=False)
    wandb.log(log_dict)
    print('Finished to train.')


def calc_dist(all_embeddings, cur_idxs):
    # print("calculate distances...")
    if len(cur_idxs) == 0:
        return []
    else:
        # cur_idxs = torch.tensor(cur_idxs)  # [Npairs, 2]
        diff = all_embeddings[cur_idxs[:, 0]] - all_embeddings[cur_idxs[:, 1]]
        distances = torch.norm(diff, p=2, dim=1)

        # # stack list of torch.tensors to one tensor as "batch"
        # diff = torch.stack(diff)

        # diff = all_embeddings[torch.tensor(cur_idxs)[:, 0], :] - all_embeddings[torch.tensor(cur_idxs)[:, 1], :]
        # distances = torch.norm(diff, p=2, dim=1)

        # diff = np.array(all_embeddings)[np.array(cur_idxs)[:, 0], :] - np.array(all_embeddings)[
        #                                                                np.array(cur_idxs)[:, 1], :]
        # distances = linalg.norm(diff,2,1)
        return distances


def evaluate(model, df_test, labels_set_dict, device, config, pos_combinations_labels, neg_combinations_labels):
    print('Starting to evaluate...')
    eval_data_set = Dataset(df_test, labels_set_dict, all_data=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_data_set, collate_fn=collate_fn, batch_size=config['batch_size'],
                                                  shuffle=False, num_workers=config['num_workers'])

    model = model.to(device)
    model.eval()

    all_labels = []
    all_embeddings = []
    print("generating embeddings...")
    t00 = timeit.default_timer()
    for step, (labels, texts_list) in enumerate(tqdm(eval_dataloader, desc="generating embeddings", leave=False)):
        labels = torch.from_numpy(np.asarray(labels)).to(device)
        all_labels.extend(labels)
        outputs = model(texts_list)
        outputs = torch.nn.functional.normalize(outputs)  # normalize
        all_embeddings.extend(outputs)
    t01 = timeit.default_timer()
    print(f"time to generate embeddings on all data is {(t01 - t00) / 60} min = {t01 - t00} sec.")

    all_embeddings = torch.stack(all_embeddings)  # [N, 512]
    t02 = timeit.default_timer()
    print(f"time to do torch.stack on all embeddings{(t02 - t01) / 60} min = {t02 - t01} sec.")
    print("creating dataset of pos neg pairs...")
    pairs_data_set = PosNegPairsDataset(pos_combinations_labels, neg_combinations_labels)
    t03 = timeit.default_timer()
    print(f"time to create dataset of pos neg pairs is {(t03 - t02) / 60} min = {t03 - t02} sec.")
    pairs_dataloader = torch.utils.data.DataLoader(pairs_data_set, batch_size=1000000, shuffle=False,
                                                   num_workers=0)  # config['num_workers'])
    total_distances = []
    total_labels = []
    print("calc distances for positive and negative pairs...")
    t1 = timeit.default_timer()
    loading_time = []
    calc_dist_time = []
    t20 = timeit.default_timer()
    for step, (pairs, labels) in enumerate(
            tqdm(pairs_dataloader, desc="calc distances for positive and negative pairs", leave=False)):
        t12 = timeit.default_timer()
        loading_time.append(t12 - t1)
        # print(f"time to load batch {(t12 - t1) / 60} min = {t12 - t1} sec.")
        distances = calc_dist(all_embeddings, pairs)
        total_distances.extend(distances)
        total_labels.extend(labels)
        t1 = timeit.default_timer()
        calc_dist_time.append(t1 - t12)
    print(f"time to load batch {np.mean(loading_time)} sec.")
    print(f"time to calc_dist {np.mean(calc_dist_time)} sec.")
    t21 = timeit.default_timer()
    print(f"time to calc distances of all is {(t21 - t20) / 60} min = {t21 - t20} sec.")
    print("finished to clac distances.")
    predictions = 1 - torch.tensor(total_distances) / 2
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(total_labels, predictions)
    roc_auc = auc(fpr, tpr)
    print(f'roc_auc = {roc_auc}\nfinished to evaluate.')
    return roc_auc
    # plt.figure()
    # lw = 2
    # plt.plot(
    #     fpr,
    #     tpr,
    #     color="darkorange",
    #     lw=lw,
    #     label="ROC curve (area = %0.2f)" % roc_auc,
    # )
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic example")
    # plt.legend(loc="lower right")
    # plt.show()


def create_correct_df(df, num_of_labels, desired_labels):  # for go-emotions dataset
    # labels_set_dict = {dmiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral}
    print("Creating corrected df...")
    start = timeit.default_timer()
    labels_set = df.columns[-num_of_labels:]
    # create new df
    list_of_labels = []
    fixed_list_of_texts = []
    for i in range(df.shape[0]):  # go over all rows
        if i == 100:  # todo:remove
            break
        if df.iloc[i, -num_of_labels - 1]:  # skip on example_very_unclear
            continue
        relevant_idxs_for_labels = np.where(df.iloc[i, -num_of_labels:].values == 1)
        if len(relevant_idxs_for_labels[0]) > 1:  # skip on multi classes example
            continue
        labels = labels_set[relevant_idxs_for_labels[0]]
        for l in labels:
            if type(desired_labels) == list:
                if l not in desired_labels:
                    continue
            try:
                fixed_list_of_texts.append(clean_text(df['text'][i]))
                list_of_labels.append(l)
            except:
                pass
    fixed_df = pd.DataFrame({'label': list_of_labels, 'text': fixed_list_of_texts})

    stop = timeit.default_timer()
    print(f'Time to create correct df is: {(stop - start) / 60} min = {stop - start} sec.')
    return fixed_df


def senity_check(df):
    love_text = []
    anger_text = []
    for i in range(df.shape[0]):
        if df.iloc[i, 0] == 'love':
            love_text.append(df.iloc[i, 1])
        elif df.iloc[i, 0] == 'anger':
            anger_text.append(df.iloc[i, 1])
    print('love text:')
    for t in love_text:
        print(t)
    print('anger text:')
    for t in anger_text:
        print(t)


def create_pos_neg_pairs_lists(df_data, tgt_file_pairs_list):
    print("creating list of positive and negative pairs.")
    classes = []
    all_pos_combinations_labels = []
    all_neg_combinations_labels = []
    set_all_labels = set(df_data['label'])
    combinations_neg = []
    for i, label in enumerate(tqdm(set_all_labels, desc="Lists creation", leave=False, position=0)):
        # print(f"Working on label {labels_idx_to_str[label]}...")
        # print("creating positive pairs list...")
        idxs_label = np.where(np.array(df_data['label']) == label)[0][:300]
        # pos
        pos_combinations_labels = list(combinations(idxs_label, 2))
        all_pos_combinations_labels.extend(pos_combinations_labels)
        classes.extend([1] * len(pos_combinations_labels))
        # neg
        # print("creating negative pairs list...")
        for j, other_label in enumerate(set_all_labels):
            # for other_class in set_all_labels:
            if other_label == label:
                continue
            if (other_label, label) in combinations_neg:
                continue
            else:
                combinations_neg.append((label, other_label))
                # print(f'{labels_idx_to_str[label]} vs {labels_idx_to_str[other_class]}')
                idxs_other_label = np.where(np.array(df_data["category"]) == other_label)[0][:20]
                neg_combinations_labels = list(product(idxs_label, idxs_other_label))
                all_neg_combinations_labels.extend(neg_combinations_labels)
                classes.extend(np.array([0] * len(neg_combinations_labels)))
    print(f"Saving pairs list to {tgt_file_pairs_list}")
    combinations_label = {'pos': all_pos_combinations_labels, 'neg': all_neg_combinations_labels}
    print(
        f"number of positive pairs = {len(all_pos_combinations_labels)}\nnumber of negative pairs = {len(all_neg_combinations_labels)}")
    tmp_dir = '/'.join(tgt_file_pairs_list.split('/')[:-1])
    print(f'tmp_dir={tmp_dir}')
    if os.path.isdir(tmp_dir):
        print("dir exist")
    else:
        print("dir does not exist")

    with open(tgt_file_pairs_list, 'wb') as fp:
        pickle.dump(combinations_label, fp)
    print('finished to save pairs list.')
    return all_pos_combinations_labels, all_neg_combinations_labels


def get_pos_neg_pairs(df_test, tgt_file_pairs_list, overwrite_pairs):
    t0 = timeit.default_timer()
    if os.path.isfile(tgt_file_pairs_list) and not overwrite_pairs:
        print("loading positvie and negative combinations pairs...")
        with open(tgt_file_pairs_list, 'rb') as fp:
            combinations_labels = pickle.load(fp)
            pos_combinations_labels = combinations_labels["pos"]
            neg_combinations_labels = combinations_labels["neg"]
        t1 = timeit.default_timer()
        print(f"time to load pairs is {(t1 - t0) / 60} min = {t1 - t0} sec.")
    else:
        pos_combinations_labels, neg_combinations_labels = create_pos_neg_pairs_lists(df_test, tgt_file_pairs_list)
        t1 = timeit.default_timer()
        print(f"time to generate pairs is {(t1 - t0) / 60} min = {t1 - t0} sec.")
    print(
        f"len of pos_combinations_labels = {len(pos_combinations_labels)}\nlen of neg_combinations_labels = {len(neg_combinations_labels)}")
    return pos_combinations_labels, neg_combinations_labels


def getting_labels_map(df_train):
    labels_set_dict = {}
    labels_idx_to_str = {}
    for i, label in enumerate(set(df_train['label'])):
        labels_set_dict[label] = i
        labels_idx_to_str[i] = label
    return labels_set_dict, labels_idx_to_str


def get_model_and_optimizer(config, path_for_loading_best_model, device):
    if config['load_model']:  # load_model
        print(f"Loading model from: {path_for_loading_best_model}")
        model = TextStyleEmbed(device=device, hidden_state_to_take=config['hidden_state_to_take'],
                               last_layer_idx_to_freeze=config['last_layer_idx_to_freeze'], scale_noise=config['scale_noise'])
        # optimizer = SGD(model.parameters(), lr=config['lr'])
        #todo: check if to take only non-frozen params:
        # optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'],
        #                 weight_decay=config['weight_decay'])
        optimizer = Adam(model.parameters(), lr=float(config['lr']))
        if 'cuda' in device.type:
            checkpoint = torch.load(path_for_loading_best_model, map_location='cuda:0')
        else:
            checkpoint = torch.load(path_for_loading_best_model, map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        #  train from scratch
        print("Train model from scratch")
        model = TextStyleEmbed(device=device, hidden_state_to_take=config['hidden_state_to_take'],
                               last_layer_idx_to_freeze=config['last_layer_idx_to_freeze'], scale_noise=config['scale_noise'])
        # optimizer = SGD(model.parameters(), lr=config['lr'])

        # take only non-frozen params:
        # optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])

        optimizer = Adam(model.parameters(), lr=float(config['lr']))
    return model, optimizer


def get_train_test_data(config, undesired_label=None):
    # load data
    if os.path.isfile(os.path.join(config['data_dir'], config['csv_file_name_train'])):
        print("Loading data...")
        df_train = pd.read_csv(os.path.join(config['data_dir'], config['csv_file_name_train']))
        print(df_train.groupby('label').size())
        df_test = pd.read_csv(os.path.join(config['data_dir'], config['csv_file_name_test']))
        unvalid_vars_train = []
        unvalid_vars_test = []
        for i, sample_data in enumerate(df_train['text']):
            if not isinstance(sample_data, str):
                # print("unvalid variable: ",sample_data)
                unvalid_vars_train.append(i)
                continue
        for j, sample_data in enumerate(df_test['text']):
            if not isinstance(sample_data, str):
                # print("unvalid variable: ",sample_data)
                unvalid_vars_test.append(j)
                continue
        df_train = df_train.drop(index=unvalid_vars_train)
        df_test = df_train.drop(index=unvalid_vars_test)

        if undesired_label:
            df_train = df_train.iloc[np.where(np.array(df_train["category"]) != undesired_label)[0], :]
            df_test = df_test.iloc[np.where(np.array(df_test["category"]) != undesired_label)[0], :]
        # df_train = df_train.iloc[:2000,:]#todo:remove
        # df_test = df_test.iloc[:2000,:]#todo:remove
    else:  # create df_train, df_test
        print("Creating data...")
        # desired_labels = ['anger','caring','optimism','love']
        # desired_labels = ['anger','love']
        # desired_labels = 'all'
        desired_labels = config['desired_labels']
        data_file = config['data_file']
        if config['dataset'] == 'flickrstyle10k':
            if type(data_file) == list:
                data = []
                for i in range(len(data_file)):
                    with open(os.path.join(config['data_dir'], data_file[i]), 'rb') as f:
                        lines = f.readlines()
                        for line in lines:
                            try:
                                normal_line = line.decode('ascii')
                            except:
                                continue
                            cleaned_text = clean_text(normal_line)
                            if not isinstance(cleaned_text, str):
                                continue
                            data.append([data_file[i].split('_')[0], cleaned_text])
            df = pd.DataFrame(data, columns=["category", "text"])
        else:  # go_emotions or twitter dataset
            if type(data_file) == list:
                print(os.path.join(config['data_dir'], data_file[0]))
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
            if config['dataset'] == 'go_emotions':
                num_of_labels = 28
                df = create_correct_df(s_df, num_of_labels, desired_labels)
            elif config['dataset'] == 'Twitter':  # change titles to Label and text
                s_df = s_df.rename(columns={'User': 'label', 'Tweet': 'text'})
                df = s_df
        print(df.head())
        if undesired_label:
            df = df.iloc[np.where(np.array(df["category"]) != undesired_label)[0], :]
        # df = df.iloc[:2000,:]#todo:remove

        print(f"Working on {config['dataset']} data. Splitting DB to train, val and test data frames.")
        df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)
        # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),  # todo check sklearn split data func - keeps proportions between classes across all splits
        #                                      [int(.8 * len(df)), int(.9 * len(df))])
        # print(len(df_train), len(df_val), len(df_test))
        print(f'len of train = {len(df_train)},len of test = {len(df_test)}')
        print(
            f"saving data file splitted to train and test sets to: {os.path.join(config['data_dir'], config['csv_file_name_train'])}\n{os.path.join(config['data_dir'], config['csv_file_name_test'])}")
        df_train.to_csv(os.path.join(config['data_dir'], config['csv_file_name_train']), index=False)
        df_test.to_csv(os.path.join(config['data_dir'], config['csv_file_name_test']), index=False)
    return df_train, df_test


def get_train_val_data(data_set_path):
    f'''

    :param data_set_path: dict. keys =   ['train', 'val', 'test'], values = path to pickl file
    :return: ds: dict:keys=['train', 'val', 'test'],values = dict:keys = list(dataset_name), values=dict:keys=key_frame,values:dict:keys=style,values=dataframe
    '''
    ds = {}
    for data_type in data_set_path:  # ['train', 'val', 'test']
        ds[data_type] = {}
        with open(data_set_path[data_type], 'rb') as r:
            data = pickle.load(r)
        for k in data:
            ds[data_type][k] = {}
            # ds[data_type][k]['factual'] = data[k]['factual']  #todo: check if there is need to concatenate factual from senticap and flickrstyle10k
            # ds[data_type][k]['img_path'] = data[k]['image_path']
            for style in data[k]:
                if style == 'img_path':
                    continue
                ds[data_type][k][style] = data[k][style]
    return ds


def convert_ds_to_df(ds, data_dir):
    df_train = None
    df_val = None
    df_test = None
    for data_type in ds:  # ['train', 'val', 'test']
        all_data = {'category': [], 'text': []}
        for k in ds[data_type]:
            for style in ds[data_type][k]:
                if style == 'image_path' or style == 'factual':
                    continue
                all_data['category'].extend([style] * len(ds[data_type][k][style]))
                all_data['text'].extend(ds[data_type][k][style])
        if data_type == 'train':
            df_train = pd.DataFrame(all_data)
            df_train.to_csv(os.path.join(data_dir, 'train.csv'))
        elif data_type == 'val':
            df_val = pd.DataFrame(all_data)
            df_val.to_csv(os.path.join(data_dir, 'val.csv'))
        elif data_type == 'test':
            df_test = pd.DataFrame(all_data)
            df_test.to_csv(os.path.join(data_dir, 'test.csv'))
    return df_train, df_val, df_test


def main():
    wandb.login(key=os.getenv('WANDB_API_KEY'))

    print('Start!')
    args = get_args()
    config = get_hparams(args)
    print(f"config_file: {config['config_file']}")
    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print(f"cur time is: {cur_time}")
    cur_date = datetime.now().strftime("%d_%m_%Y")
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config['cuda_idx_num'])

    np.random.seed(112)  # todo there may be many more seeds to fix
    torch.cuda.manual_seed(112)
    # overwrite_pairs = True  # todo

    checkpoints_dir = os.path.join(os.path.expanduser('~'), 'checkpoints')
    global_dir_name_for_save_models = os.path.join(checkpoints_dir, config['global_dir_name_for_save_models'])
    if not os.path.isdir(global_dir_name_for_save_models):
        os.makedirs(global_dir_name_for_save_models)
    experiment_dir_date = os.path.join(checkpoints_dir, config['global_dir_name_for_save_models'], cur_date)
    if not os.path.isdir(experiment_dir_date):
        os.makedirs(experiment_dir_date)
    experiment_dir = os.path.join(checkpoints_dir, config['global_dir_name_for_save_models'], cur_date, cur_time)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    data_dir = os.path.join(os.path.expanduser('~'), 'data')

    data_set_path = {'train': {}, 'val': {}, 'test': {}}
    for data_type in ['train', 'val', 'test']:
        data_set_path[data_type] = os.path.join(data_dir, config['dataset'], 'annotations',
                                                             data_type + '.pkl')

    path_for_saving_last_model = os.path.join(experiment_dir, config['txt_embed_model_name'])
    path_for_saving_best_model = os.path.join(experiment_dir, config['txt_embed_best_model_name'])

    if 'path_for_loading_best_model' in config and config['path_for_loading_best_model']:
        path_for_loading_best_model = config['path_for_loading_best_model']
    else:
        path_for_loading_best_model = os.path.join(checkpoints_dir, 'best_models', config['dataset'], config['best_model_name'])


    if config['plot_only_clustering']:
        tgt_file_vec_emb = {
            'mean': os.path.join(checkpoints_dir, 'best_models', config['txt_embed_mean_vec_emb_file']),
            'std':  os.path.join(checkpoints_dir, 'best_models', config['txt_embed_std_vec_emb_file'])}
    else:
        tgt_file_vec_emb = {'mean': os.path.join(experiment_dir, config['txt_embed_mean_vec_emb_file']),
                            'std':  os.path.join(experiment_dir, config['txt_embed_std_vec_emb_file'])}
    # tgt_file_pairs_list = os.path.join(config['data_dir'],config['tgt_file_pairs_list'])

    use_cuda = torch.cuda.is_available()
    # device = torch.device(f"cuda:{config['desired_cuda_num']}" if use_cuda else "cpu")  # todo: remove
    device = torch.device("cuda" if use_cuda else "cpu")  # todo: remove
    wandb.init(project='text_style_embedding',
               config=config,
               resume=config['resume'],
               id=config['run_id'],
               mode=config['wandb_mode'], #disabled, offline, online'
               tags=config['tags'])

    ds = get_train_val_data(data_set_path)
    df_train, df_val, df_test = convert_ds_to_df(ds, data_dir)
    print(len(df_train), len(df_val), len(df_test))
    print(f"labels: {config['labels_set_dict']}")

    # df_train,df_test = get_train_test_data(config, config['undesired_label'])
    model, optimizer = get_model_and_optimizer(config, path_for_loading_best_model, device)
    # config['labels_set_dict'],config['labels_idx_to_str'] = getting_labels_map(df_train)
    if config['plot_only_clustering']:
        print("********plot_only_clustering********")
        log_dict = {}
        train_data_set, val_data_set, test_data_set = Dataset(df_train, config['labels_set_dict']), Dataset(df_val, config['labels_set_dict']), Dataset(df_test, config['labels_set_dict'])
        train_dataloader = torch.utils.data.DataLoader(train_data_set, collate_fn=collate_fn,
                                                       batch_size=config['batch_size'], shuffle=True,
                                                       num_workers=config['num_workers'])
        val_dataloader = torch.utils.data.DataLoader(val_data_set, collate_fn=collate_fn,
                                                     batch_size=config['batch_size'],
                                                     shuffle=True, num_workers=config['num_workers'])
        test_dataloader = torch.utils.data.DataLoader(test_data_set, collate_fn=collate_fn,
                                                     batch_size=config['batch_size'],
                                                     shuffle=True, num_workers=config['num_workers'])
        log_dict["final/final_train_plot"] = plot_final_graph_after_training(device, config, path_for_loading_best_model,
                                                                       config['labels_idx_to_str'], tgt_file_vec_emb, df_train,
                                                                       train_dataloader, title="Train")

        log_dict["final/final_val_plot"] = plot_final_graph_after_training(device, config, path_for_loading_best_model,
                                                                     config['labels_idx_to_str'], tgt_file_vec_emb, df_val,
                                                                     val_dataloader, title="Val")
        log_dict["final/final_test_plot"] = plot_final_graph_after_training(device, config, path_for_loading_best_model,
                                                                     config['labels_idx_to_str'], tgt_file_vec_emb, df_test,
                                                                     test_dataloader, title="Test")
        print("send data to wandb...")
        wandb.log(log_dict)

        # log_dict_train = plot_graph_on_all_data(df_train,
        #                                         config['labels_set_dict'], config['labels_idx_to_str'], device, model,
        #                                         config['inner_batch_size'], config['batch_size'],
        #                                         "final embedding on trainning set for best model",
        #                                         tgt_file_vec_emb, True, True, config['num_workers'],
        #                                         config['desired_labels'])
        print('finished to plot clustering for the best model.')
        exit(0)
    # senity_check(df_train)
    # pos_combinations_labels, neg_combinations_labels = get_pos_neg_pairs(df_test, tgt_file_pairs_list, overwrite_pairs)

    train(model, optimizer, df_train, df_val, config['labels_set_dict'], config['labels_idx_to_str'], path_for_saving_last_model,
          path_for_saving_best_model, device, tgt_file_vec_emb, config)
    # evaluate(model,  filtered_df_test, config['labels_set_dict'], device, config,pos_combinations_labels,neg_combinations_labels)

    print('  finish!')


if __name__ == '__main__':
    main()
