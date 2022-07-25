from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
from numpy import linalg
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
from itertools import combinations,product
import clip
from sklearn.metrics import roc_curve,auc
import os
import random
from sklearn.model_selection import train_test_split
import timeit

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class PosNegPairsDataset(torch.utils.data.Dataset):
    def __init__(self,pos_combinations_labels,neg_combinations_labels):
        self.pairs = torch.tensor(pos_combinations_labels + neg_combinations_labels)
        self.labels = torch.tensor([1]*len(pos_combinations_labels) + [0]*len(neg_combinations_labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.pairs[item],self.labels[item]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labels_set_dict, inner_batch_size=1,all_data=False):
        self.labels = [labels_set_dict[label] for label in df['label']] #create list of idxs for labels
        self.labels_set = list(set(self.labels))
        self.texts = list(df['text'])#df['Tweet'] #[text for text in df['Tweet']]
        self.batch_size_per_label = inner_batch_size
        self.all_data = all_data #boolean
        pass

    def classes(self):
        return self.labels

    def __len__(self):
        if self.all_data:
            return len(self.labels)
        else:#get samples from set of labels
            return len(self.labels_set)


    def __getitem__(self, item):
        if self.all_data:
            label = self.labels[item]
            text = self.texts[item]
            return text, label

        else: # get samples from data
            label = self.labels_set[item]
            list_idxs_for_label = np.array(self.labels) == label
            full_tweets_list = list(operator.itemgetter(list_idxs_for_label)(np.array(self.texts)))
            batch_tweets = random.sample(full_tweets_list,min(len(full_tweets_list),self.batch_size_per_label))
            return batch_tweets, label


#todo: remove comment
# class TextStyleEmbed(nn.Module):
#
#     def __init__(self, dropout=0.05):
#         super(TextStyleEmbed, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         for param in self.bert.parameters():
#             param.requires_grad_(True)
#         self.dropout = nn.Dropout(dropout)
#         self.linear1 = nn.Linear(768, 128)
#         self.linear2 = nn.Linear(700, 600)
#         self.relu = nn.ReLU()
#
#     def forward(self, input_id, mask):
#         _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False) # pooled_output is the embedding token of the [CLS] token for all batch
#         #dropout_output = self.dropout(pooled_output)
#         relu_output = self.relu(pooled_output)
#         linear1_output = self.linear1(relu_output)
#         #relu_output = self.relu(linear1_output)
#         #linear2_output = self.linear2(relu_output)
#         output = torch.nn.functional.normalize(linear1_output)
#         #output = torch.nn.functional.normalize(pooled_output)
#         return output




# with torch.no_grad():
#     text_style_features = clip.tokenize(text).to(device)
class TextStyleEmbed(nn.Module):

    def __init__(self, dropout=0.05,device=torch.device('cpu')):
        super(TextStyleEmbed, self).__init__()
        # Initialize CLIP
        clip_checkpoints = './clip_checkpoints'
        self.clip, clip_preprocess = clip.load("ViT-B/32", device=device,
                                          download_root=clip_checkpoints, jit=False)  # clip_preprocess for images
        self.device = device
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()

    def forward(self, text):
        # clip_texts = clip.tokenize(text, truncate=True).to(self.device)
        # text_features = self.clip.encode_text(clip_texts)
        inputs = clip.tokenize(text, truncate=True).to(self.device)
        x = self.clip.encode_text(inputs)
        x = self.linear1(x.float())
        x = self.relu(x)
        x = self.linear2(x)
        text_features = x / x.norm(dim=-1, keepdim=True)
        return text_features


# def collate_fn(data): #for bert
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
#     tokenized_texts_list = tokenizer(texts_list, padding='max_length', max_length=40, truncation=True,
#                                      return_tensors="pt")
#     return tokenized_texts_list, labels_list, texts_list


def collate_fn(data): #for clip
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
    return labels_list, texts_list

def get_auc(all_embeddings,pos_combinations_labels, neg_combinations_labels):
    t01 = timeit.default_timer()
    # all_embeddings = torch.stack(all_embeddings)  # [N, 512]
    t02 = timeit.default_timer()
    print(f"time to do torch.stack on all embeddings{(t02 - t01)/60} min = {t02 - t01} sec.")
    print("creating dataset of pos neg pairs...")
    pairs_data_set = PosNegPairsDataset(pos_combinations_labels, neg_combinations_labels)
    t03 = timeit.default_timer()
    print(f"time to create dataset of pos neg pairs is {(t03 - t02)/60} min = {t03 - t02} sec.")
    pairs_dataloader = torch.utils.data.DataLoader(pairs_data_set, batch_size=1000000, shuffle=False,
                                                  num_workers=0)  #config['num_workers'])
    total_distances = []
    total_labels = []
    print("calc distances for positive and negative pairs...")
    t1 = timeit.default_timer()
    loading_time = []
    calc_dist_time = []
    t20 = timeit.default_timer()
    for step, (pairs,labels) in enumerate(tqdm(pairs_dataloader, desc="calc distances for positive and negative pairs", leave=False)):
        t12 = timeit.default_timer()
        loading_time.append(t12 - t1)
        # print(f"time to load batch {(t12 - t1) / 60} min = {t12 - t1} sec.")
        distances = calc_dist(all_embeddings,pairs)
        total_distances.extend(distances)
        total_labels.extend(labels)
        t1 = timeit.default_timer()
        calc_dist_time.append(t1-t12)
    print(f"time to load batch {np.mean(loading_time)} sec.")
    print(f"time to calc_dist {np.mean(calc_dist_time)} sec.")
    t21 = timeit.default_timer()
    print(f"time to calc distances of all is {(t21 - t20) / 60} min = {t21 - t20} sec.")
    print("finished to clac distances.")
    predictions = 1-torch.tensor(total_distances)/2
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(total_labels, predictions)
    roc_auc = auc(fpr, tpr)
    print(f'roc_auc = {roc_auc}\nfinished to evaluate.')
    return roc_auc

def plot_graph_on_all_data(df_data, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, batch_size, title,tgt_file_vec_emb, all_data=False,save_vec_emb = False,num_workers=0,desired_labels = 'all',pos_combinations_labels=None, neg_combinations_labels = None):
    print(f"Plotting graph for {title} for {desired_labels} labels...")
    if desired_labels == 'all':
        desired_df = df_data
    else:
        desired_df = pd.DataFrame()
        for l in desired_labels:
            l_df = df_data.iloc[np.where(np.array(df_data["label"]) == l)[0], :]
            desired_df = pd.concat([desired_df,l_df])
    data_set = Dataset(desired_df, labels_set_dict, inner_batch_size, all_data)
    eval_dataloader = torch.utils.data.DataLoader(data_set, collate_fn=collate_fn, batch_size = batch_size, shuffle=True,
                                                   num_workers=num_workers)
    model.to(device)
    model.eval()

    log_dict = {}
    total_outputs = []
    total_labels = []
    total_texts_list = []
    with torch.no_grad():
        # for step, (tokenized_texts_list, labels, texts_list) in enumerate(
        #         tqdm(eval_dataloader, desc="Evaluation", leave=False)):
        for step, (labels, texts_list) in enumerate(pbar := tqdm(eval_dataloader, desc="Evaluation", leave=False)):
            # labels = torch.from_numpy(np.asarray(labels)).to(device)
            total_labels.extend(labels)
            outputs = model(texts_list)
            # outputs = torch.nn.functional.normalize(outputs)  # normalize# todo: check if need to do that
            outputs = outputs.to(torch.device('cpu'))
            total_outputs.extend(outputs)
            total_texts_list.extend(texts_list)
        total_outputs = torch.stack(total_outputs)
        total_labels_str = [labels_idx_to_str[user_idx] for user_idx in total_labels]
        if save_vec_emb:
            print("Calculate mean and median embedding vectors...")
            mean_embedding_vectors_to_save = {}
            median_embedding_vectors_to_save = {}
            for label in set(desired_df["label"]):
                vectors_embedding = total_outputs[np.where(np.array(total_labels_str) == label), :]
                # median_vector_embedding = np.median(vectors_embedding[0], 0)
                #median_embedding_vectors_to_save[label] = median_vector_embedding
                mean_vector_embedding = torch.mean(vectors_embedding[0], 0)
                mean_embedding_vectors_to_save[label] = mean_vector_embedding
            print('Saving mean of embedding vectors to: '+tgt_file_vec_emb['mean']+'...')
            with open(tgt_file_vec_emb['mean'], 'wb') as fp:
                pickle.dump(mean_embedding_vectors_to_save, fp)
            #print(f'Saving median of embedding vectors to: '+tgt_file_vec_emb['median']+'...')
            #with open(tgt_file_vec_emb['median'], 'wb') as fp:
            #    pickle.dump(median_embedding_vectors_to_save, fp)
            print(f'Finished to save.')

            print('print for wandb')
            # variables with mean
            total_outputs_with_representation = total_outputs.to(torch.device('cpu'))
            for label in set(desired_df["label"]):
                # #insert mean and median to the beginning
                # total_labels_str = [f'mean_{label}',f'median_{label}']+total_labels_str
                # total_texts_list = [f'mean_{label}', f'median_{label}'] + total_texts_list
                # total_outputs_with_representation = np.concatenate(
                #     (np.array([mean_embedding_vectors_to_save[label]]),
                #      np.array([median_embedding_vectors_to_save[label]]), total_outputs_with_representation), axis=0)
                # insert mean and median to the beginning
                total_labels_str = [f'mean_{label}'] + total_labels_str
                total_texts_list = [f'mean_{label}'] + total_texts_list
                # total_outputs_with_representation = np.concatenate(
                #     (np.array(mean_embedding_vectors_to_save[label]), total_outputs_with_representation), axis=0)
                total_outputs_with_representation = np.concatenate(
                    (np.expand_dims(np.array(mean_embedding_vectors_to_save[label]),axis=0), total_outputs_with_representation), axis=0)
            total_outputs = total_outputs_with_representation
        labeldf = pd.DataFrame({'Label': total_labels_str})
        embdf = pd.DataFrame(total_outputs, columns=[f'emb{i}' for i in range(total_outputs.shape[1])])
        textdf = pd.DataFrame({'text': total_texts_list})
        all_data = pd.concat([labeldf, embdf, textdf], axis=1, ignore_index=True)
        all_data.columns = ['Label'] + [f'emb{i}' for i in range(total_outputs.shape[1])] + ['text']
        log_dict[title] = all_data
        cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
        print(f"cur time is: {cur_time}")
        print("send data to wb")
        t1 = timeit.default_timer()
        wandb.log({title: all_data})  # todo log without commit
        t2 = timeit.default_timer()
        print(f'Time to send to wb is: ={(t2-t1)/60} min. = {t2-t1} sec.')
    if pos_combinations_labels:
        roc_auc = get_auc(total_outputs, pos_combinations_labels, neg_combinations_labels)
        return log_dict, roc_auc
    return log_dict


def train(model, optimizer, df_train, df_test, labels_set_dict, labels_idx_to_str, path_for_saving_last_model, path_for_saving_best_model, device, tgt_file_vec_emb, config,pos_combinations_labels, neg_combinations_labels, **kwargs):
    # some_var = kwconfig['get('some_var_name', None)  # todo function call: train(train_args, some_var_name=some_var)
    # val_batch_size_for_plot = len(set(df_test['label'])) #min(batch_size,len(set(df_test['label'])))# suppose that the first column is for label
    train_batch_size_for_plot = len(set(df_train['label'])) #min(batch_size,len(set(df_train['label'])))
    val_batch_size_for_plot = len(set(df_test['label'])) #min(batch_size,len(set(df_train['label'])))

    train_data_set = Dataset(df_train, labels_set_dict, config['inner_batch_size'],True)
    train_dataloader = torch.utils.data.DataLoader(train_data_set, collate_fn=collate_fn, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    print("Sanity check on train df...")
    log_dict_train = plot_graph_on_all_data(df_train.iloc[np.arange(0, min(5000,len(df_train)), 50),:], labels_set_dict, labels_idx_to_str, device, model,
                                            config['inner_batch_size'],
                                            train_batch_size_for_plot, "sanity_check_initial_train",
                                            tgt_file_vec_emb, True, False,config['num_workers'])

    print('Starting to train...')
    model = model.to(device)
    best_loss = 1e16
    last_best_epoch = -11
    for epoch in range(config['epochs']):
        t000 = timeit.default_timer()
        model.train()
        running_loss = []
        # for step, (tokenized_texts_list, labels, texts_list) in enumerate(pbar:= tqdm(train_dataloader, desc="Training", leave=False,)):
        for step, (labels, texts_list) in enumerate(pbar := tqdm(train_dataloader, desc="Training", leave=False)):
            labels = torch.from_numpy(np.asarray(labels)).to(device)
            outputs = model(texts_list)
            outputs = torch.nn.functional.normalize(outputs)  # normalize

            # triplet loss
            loss, num_positive_triplets, num_valid_triplets, all_triplet_loss_avg = mining.online_mine_all(labels, outputs, config['margin'], device=device)
            fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

            loss.backward()
            optimizer.step()

            # running_loss.append(loss.cpu().detach().numpy())
            running_loss.append(all_triplet_loss_avg)
        t001 = timeit.default_timer()
        print(f"time for single epoch is {(t001 - t000) / 60} min = {t001 - t000} sec.")
        avg_loss = np.mean([loss_elem.item() for loss_elem in running_loss])
        pbar.set_description("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, config['epochs'], avg_loss))
        # print("\nEpoch: {}/{} - Loss: {:.4f}".format(epoch + 1, config['epochs'], all_triplet_loss_avg),'\n')
        log_dict = {'train/epoch': epoch,
                    'train/train_loss': loss.cpu().detach().numpy(),
                    'train/fraction_positive_triplets': fraction_positive_triplets,
                    'train/num_positive_triplets': num_positive_triplets,
                    'train/all_triplet_loss_avg': all_triplet_loss_avg}
        wandb.log({"log_dict": log_dict})
        if np.mod(epoch, 10) == 0:  # plot on every 10 epochs
            # save model
            print(f'Epoch = {epoch},Saving model to: {path_for_saving_last_model}...')
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        }, path_for_saving_last_model)  # finally check on all data training
            log_dict_train = plot_graph_on_all_data(df_train.iloc[np.arange(0, min(5000,len(df_train)), 50),:], labels_set_dict, labels_idx_to_str, device, model,
                                                    config['inner_batch_size'],train_batch_size_for_plot, "train_text", tgt_file_vec_emb, True, False, config['num_workers'])
            # log_dict_val, roc_auc = plot_graph_on_all_data(df_test.iloc[np.arange(0, min(15000,len(df_test)), 50),:], labels_set_dict, labels_idx_to_str, device, model,
            log_dict_val, roc_auc = plot_graph_on_all_data(df_test, labels_set_dict, labels_idx_to_str, device, model,
                                                  config['inner_batch_size'],val_batch_size_for_plot, "val_text", tgt_file_vec_emb,
                                                  True, False, config['num_workers'],pos_combinations_labels=pos_combinations_labels,
                                                       neg_combinations_labels = neg_combinations_labels)
            # log_dict = {**log_dict, **log_dict_train, **log_dict_val}
            # wandb.log({"log_dict": log_dict})
            # todo - with every log save the latest model (so we can resume training from the same point.)
            # roc_auc = evaluate(model, df_test, labels_set_dict, device, config, pos_combinations_labels,
            #          neg_combinations_labels)
            log_dict = {'train/epoch': epoch,
                        'train/train_loss': loss.cpu().detach().numpy(),
                        'train/fraction_positive_triplets': fraction_positive_triplets,
                        'train/num_positive_triplets': num_positive_triplets,
                        'train/all_triplet_loss_avg': all_triplet_loss_avg,
                        'train/roc_auc': roc_auc}
            wandb.log({"log_dict": log_dict})

        if avg_loss<best_loss:
            print(f'Loss is improved. Epoch = {epoch}, Saving the better model to: {path_for_saving_best_model}...')
            best_loss = avg_loss
            # save model
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        }, path_for_saving_best_model)  # finally check on all data training
            if epoch>last_best_epoch+50:
                last_best_epoch = epoch
                # log_dict_train = plot_graph_on_all_data(df_train, labels_set_dict, labels_idx_to_str, device, model,
                #                                         config['inner_batch_size'], train_batch_size_for_plot, "train_text_for_best_model",
                #                                         tgt_file_vec_emb, True, True, config['num_workers'])
                # log_dict_val = plot_graph_on_all_data(df_test.iloc[np.arange(0, min(15000,len(df_train)), 50),:], labels_set_dict, labels_idx_to_str, device, model,
                log_dict_val = plot_graph_on_all_data(df_test, labels_set_dict, labels_idx_to_str, device, model,
                                                      config['inner_batch_size'], val_batch_size_for_plot, "val_text",
                                                      tgt_file_vec_emb,
                                                      True, False, config['num_workers'])
                # log_dict = {**log_dict, **log_dict_train, **log_dict_val}

    # plot_graph_on_all_data(df_train, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, train_batch_size_for_plot,"final_train_text",wb, tgt_file_vec_emb)
    model = TextStyleEmbed(device=device)
    optimizer = SGD(model.parameters(), lr=config['lr'])
    checkpoint = torch.load(path_for_saving_best_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Finished to train, plotting final embedding of entire training set")
    log_dict_train = plot_graph_on_all_data(df_train, labels_set_dict, labels_idx_to_str, device, model,
                                            config['inner_batch_size'], config['batch_size'], "final embedding on trainning set_for_best_model",
                                            tgt_file_vec_emb, True, True, config['num_workers'])
    print('Finished to train.')


def calc_dist(all_embeddings, cur_idxs):
    # print("calculate distances...")
    if len(cur_idxs)==0:
        return []
    else:
        # cur_idxs = torch.tensor(cur_idxs)  # [Npairs, 2]
        diff = all_embeddings[cur_idxs[:,0]] - all_embeddings[cur_idxs[:,1]]
        distances = torch.norm(diff, p=2, dim=1)

        # # stack list of torch.tensors to one tensor as "batch"
        # diff = torch.stack(diff)

        # diff = all_embeddings[torch.tensor(cur_idxs)[:, 0], :] - all_embeddings[torch.tensor(cur_idxs)[:, 1], :]
        # distances = torch.norm(diff, p=2, dim=1)


        # diff = np.array(all_embeddings)[np.array(cur_idxs)[:, 0], :] - np.array(all_embeddings)[
        #                                                                np.array(cur_idxs)[:, 1], :]
        # distances = linalg.norm(diff,2,1)
        return distances


def evaluate(model, df_test, labels_set_dict, device, config,pos_combinations_labels,neg_combinations_labels):
    print('Starting to evaluate...')
    eval_data_set = Dataset(df_test, labels_set_dict,all_data=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_data_set, collate_fn=collate_fn, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

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
        outputs = torch.nn.functional.normalize(outputs) # normalize
        all_embeddings.extend(outputs)
    t01 = timeit.default_timer()
    print(f"time to generate embeddings on all data is {(t01 - t00)/60} min = {t01 - t00} sec.")

    all_embeddings = torch.stack(all_embeddings)  # [N, 512]
    t02 = timeit.default_timer()
    print(f"time to do torch.stack on all embeddings{(t02 - t01)/60} min = {t02 - t01} sec.")
    print("creating dataset of pos neg pairs...")
    pairs_data_set = PosNegPairsDataset(pos_combinations_labels, neg_combinations_labels)
    t03 = timeit.default_timer()
    print(f"time to create dataset of pos neg pairs is {(t03 - t02)/60} min = {t03 - t02} sec.")
    pairs_dataloader = torch.utils.data.DataLoader(pairs_data_set, batch_size=1000000, shuffle=False,
                                                  num_workers=0)  #config['num_workers'])
    total_distances = []
    total_labels = []
    print("calc distances for positive and negative pairs...")
    t1 = timeit.default_timer()
    loading_time = []
    calc_dist_time = []
    t20 = timeit.default_timer()
    for step, (pairs,labels) in enumerate(tqdm(pairs_dataloader, desc="calc distances for positive and negative pairs", leave=False)):
        t12 = timeit.default_timer()
        loading_time.append(t12 - t1)
        # print(f"time to load batch {(t12 - t1) / 60} min = {t12 - t1} sec.")
        distances = calc_dist(all_embeddings,pairs)
        total_distances.extend(distances)
        total_labels.extend(labels)
        t1 = timeit.default_timer()
        calc_dist_time.append(t1-t12)
    print(f"time to load batch {np.mean(loading_time)} sec.")
    print(f"time to calc_dist {np.mean(calc_dist_time)} sec.")
    t21 = timeit.default_timer()
    print(f"time to calc distances of all is {(t21 - t20) / 60} min = {t21 - t20} sec.")
    print("finished to clac distances.")
    predictions = 1-torch.tensor(total_distances)/2
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




def create_correct_df(df,num_of_labels,desired_labels):
    # labels_set_dict = {dmiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral}
    print("Creating corrected df...")
    start = timeit.default_timer()
    labels_set = df.columns[-num_of_labels:]
    #create new df
    list_of_labels = []
    fixed_list_of_texts = []
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
                fixed_list_of_texts.append(clean_text(df['text'][i]))
                list_of_labels.append(l)
            except:
                pass
    fixed_df = pd.DataFrame({'label': list_of_labels, 'text': fixed_list_of_texts})

    stop = timeit.default_timer()
    print(f'Time to create correct df is: {(stop - start)/60} min = {stop - start} sec.')
    return fixed_df

def senity_check(df):
    love_text = []
    anger_text = []
    for i in range(df.shape[0]):
        if df.iloc[i, 0]=='love':
            love_text.append(df.iloc[i, 1])
        elif df.iloc[i, 0]=='anger':
            anger_text.append(df.iloc[i, 1])
    print('love text:')
    for t in love_text:
        print(t)
    print('anger text:')
    for t in anger_text:
        print(t)

def create_pos_neg_pairs_lists(df_data,tgt_file_pairs_list):
    print("creating list of positive and negative pairs.")
    classes = []
    all_pos_combinations_labels = []
    all_neg_combinations_labels = []
    set_all_labels = set(df_data['label'])
    combinations_neg = []
    for i,label in enumerate(tqdm(set_all_labels, desc="Lists creation", leave=False, position=0)):
        # print(f"Working on label {labels_idx_to_str[label]}...")
        # print("creating positive pairs list...")
        idxs_label = np.where(np.array(df_data['label']) == label)[0][:300]
        # pos
        pos_combinations_labels = list(combinations(idxs_label,2))
        all_pos_combinations_labels.extend(pos_combinations_labels)
        classes.extend([1]*len(pos_combinations_labels))
        #neg
        # print("creating negative pairs list...")
        for j, other_label in enumerate(set_all_labels):
        # for other_class in set_all_labels:
            if other_label == label:
                continue
            if (other_label, label) in combinations_neg:
                continue
            else:
                combinations_neg.append((label,other_label))
                # print(f'{labels_idx_to_str[label]} vs {labels_idx_to_str[other_class]}')
                idxs_other_label = np.where(np.array(df_data["label"]) == other_label)[0][:20]
                neg_combinations_labels = list(product(idxs_label, idxs_other_label))
                all_neg_combinations_labels.extend(neg_combinations_labels)
                classes.extend(np.array([0] * len(neg_combinations_labels)))
    print(f"Saving pairs list to {tgt_file_pairs_list}")
    combinations_label = {'pos': all_pos_combinations_labels, 'neg': all_neg_combinations_labels}
    print(f"number of positive pairs = {len(all_pos_combinations_labels)}\nnumber of negative pairs = {len(all_neg_combinations_labels)}")
    with open(tgt_file_pairs_list, 'wb') as fp:
        pickle.dump(combinations_label, fp)
    print('finished to save pairs list.')
    return all_pos_combinations_labels,all_neg_combinations_labels


def get_pos_neg_pairs(df_test,tgt_file_pairs_list,overwrite_pairs):
    t0 = timeit.default_timer()
    if os.path.isfile(tgt_file_pairs_list) and not overwrite_pairs:
        print("loading positvie and negative combinations pairs...")
        with open(tgt_file_pairs_list, 'rb') as fp:
            combinations_labels = pickle.load(fp)
            pos_combinations_labels = combinations_labels["pos"]
            neg_combinations_labels = combinations_labels["neg"]
        t1 = timeit.default_timer()
        print(f"time to load pairs is {(t1 - t0)/60} min = {t1 - t0} sec.")
    else:
        pos_combinations_labels,neg_combinations_labels = create_pos_neg_pairs_lists(df_test, tgt_file_pairs_list)
        t1 = timeit.default_timer()
        print(f"time to generate pairs is {(t1 - t0)/60} min = {t1 - t0} sec.")
    print(f"len of pos_combinations_labels = {len(pos_combinations_labels)}\nlen of neg_combinations_labels = {len(neg_combinations_labels)}")
    return pos_combinations_labels,neg_combinations_labels

def getting_labels_map(config,df_train):
    labels_set_dict = {}
    labels_idx_to_str = {}
    if config['data_name'] == 'go_emotions':
        for i, label in enumerate(set(df_train['label'])):
            labels_set_dict[label] = i
            labels_idx_to_str[i] = label
    elif config['data_name']=='Twitter':
        for i, label in enumerate(set(df_train['label'])):
            labels_set_dict[label] = i
            labels_idx_to_str[i] = label
    return labels_set_dict,labels_idx_to_str

def get_model_and_optimizer(config, path_for_loading_best_model, device):
    if config['resume']: #load_model
        print(f"Loading model from: {path_for_loading_best_model}")
        model = TextStyleEmbed(device=device)
        optimizer = SGD(model.parameters(), lr=config['lr'])
        checkpoint = torch.load(path_for_loading_best_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        #  train from scratch
        print("Train model from scratch")
        model = TextStyleEmbed(device=device)
        optimizer = SGD(model.parameters(), lr=config['lr'])
    return model,optimizer

def get_train_test_data(config, undesired_label = None):
    #load data
    if os.path.isfile(os.path.join(config['data_dir'],config['csv_file_name_train'])):
        df_train = pd.read_csv(os.path.join(config['data_dir'],config['csv_file_name_train']))
        print(df_train.groupby('label').size())
        df_test = pd.read_csv(os.path.join(config['data_dir'],config['csv_file_name_test']))
        if undesired_label:
            df_train = df_train.iloc[np.where(np.array(df_train["label"]) != undesired_label)[0], :]
            df_test = df_test.iloc[np.where(np.array(df_test["label"]) != undesired_label)[0], :]
        # df_train = df_train.iloc[:2000,:]#todo:remove
        # df_test = df_test.iloc[:2000,:]#todo:remove
    else: #create df_train, df_test
        # desired_labels = ['anger','caring','optimism','love']
        # desired_labels = ['anger','love']
        # desired_labels = 'all'
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
            df = create_correct_df(s_df, num_of_labels, desired_labels)
        elif config['data_name'] == 'Twitter': # change titles to Label and text
            s_df = s_df.rename(columns={'User': 'label', 'Tweet': 'text'})
            df = s_df

        if undesired_label:
            df = df.iloc[np.where(np.array(df["label"]) != undesired_label)[0], :]
        # df = df.iloc[:2000,:]#todo:remove

        print(f"Working on {config['data_name']} data. Splitting DB to train, val and test data frames.")
        df_train, df_test = train_test_split(df, test_size = 0.15, random_state = 42)
        # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),  # todo check sklearn split data func - keeps proportions between classes across all splits
        #                                      [int(.8 * len(df)), int(.9 * len(df))])
        # print(len(df_train), len(df_val), len(df_test))
        print(f'len of train = {len(df_train)},len of test = {len(df_test)}')
        print(f"saving data file splitted to train and test sets to: {os.path.join(config['data_dir'],config['csv_file_name_train'])}\n{os.path.join(config['data_dir'],config['csv_file_name_test'])}")
        df_train.to_csv(os.path.join(config['data_dir'],config['csv_file_name_train']))
        df_test.to_csv(os.path.join(config['data_dir'],config['csv_file_name_test']))
    return df_train,df_test



def main():
    desired_cuda_num = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = desired_cuda_num
    np.random.seed(112)  # todo there may be many more seeds to fix
    torch.cuda.manual_seed(112)
    overwrite_pairs = False#todo

    print('Start!')
    args = parser.parse_args()
    config = get_hparams(args)
    checkpoints_dir = config['checkpoints_dir']
    experiment_name = config['experiment_name']
    if experiment_name=='cur_time':
        experiment_name = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print('experiment name is: '+experiment_name)
    experiment_dir = os.path.join(checkpoints_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    elif not config['override']:
        r = None
        while r not in ['y', 'n']:
            r = input(f'{experiment_dir} already exists, do you want to override? (y/n)')
            if r == 'n':
                exit(0)
            elif r == 'y':
                print('overriding results in ', experiment_dir)
                break

    path_for_saving_last_model = os.path.join(experiment_dir, config['model_name'])
    path_for_saving_best_model = os.path.join(experiment_dir, config['best_model_name'])
    path_for_loading_best_model = os.path.join(checkpoints_dir, 'best_model', config['best_model_name'])
    tgt_file_vec_emb = {'mean': os.path.join(experiment_dir, config['mean_vec_emb_file']),
                        'median': os.path.join(experiment_dir, config['median_vec_emb_file'])}
    tgt_file_pairs_list = os.path.join(config['data_dir'],config['tgt_file_pairs_list'])

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{desired_cuda_num}" if use_cuda else "cpu")#todo: remove
    wandb.init(project='zero-shot-learning',
               config=config,
               resume=config['resume'],
               id=config['run_id'],
               mode=config['wandb_mode'],
               tags=config['tags'])
    df_train,df_test = get_train_test_data(config, config['undesired_label'])
    model,optimizer = get_model_and_optimizer(config,path_for_loading_best_model,device)
    labels_set_dict,labels_idx_to_str = getting_labels_map(config,df_train)
    if config['plot_only_clustering']:
        log_dict_train = plot_graph_on_all_data(df_train,
                                                labels_set_dict, labels_idx_to_str, device, model,
                                                config['inner_batch_size'], config['batch_size'],
                                                "final embedding on trainning set for best model",
                                                tgt_file_vec_emb, True, True, config['num_workers'],
                                                config['desired_labels'])
        print('finished to plot clustering for the best model.')
        exit(0)
    # senity_check(df_train)
    pos_combinations_labels, neg_combinations_labels = get_pos_neg_pairs(df_test, tgt_file_pairs_list, overwrite_pairs)
    train(model, optimizer, df_train, df_test, labels_set_dict, labels_idx_to_str, path_for_saving_last_model,path_for_saving_best_model, device, tgt_file_vec_emb,config,pos_combinations_labels, neg_combinations_labels)
    # evaluate(model,  filtered_df_test, labels_set_dict, device, config,pos_combinations_labels,neg_combinations_labels)

    print('  finish!')


if __name__ == '__main__':
    main()

    # references:
    # https://github.com/avrech/learning2cut/blob/master/experiments/cut_selection_dqn/default_parser.py
    # https://github.com/avrech/learning2cut