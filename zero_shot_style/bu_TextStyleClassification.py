######################
# maybe try to use https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
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
#from zero_shot_style.model import mining
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
# from pytorch_metric_learning import losses
# losses.NPairsLoss(**kwargs)
NUM_OF_LABELS = 2#4#5
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


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
        self.labels = [labels_set_dict[label] for label in df["label"]] #create list of idxs for labels
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
            label_vec = np.zeros(NUM_OF_LABELS)
            label_vec[label] = 1
            text = self.texts[item]
            return text, label_vec

        else: # get samples from data
            label = self.labels_set[item]
            a = np.array([label]*self.batch_size_per_label)
            label_vec = np.zeros((a.size,NUM_OF_LABELS))
            label_vec[np.arange(a.size), a] = 1

            #label_vec = np.zeros(NUM_OF_LABELS)
            #label_vec[label] = 1
            list_idxs_for_label = np.array(self.labels) == label
            full_tweets_list = list(operator.itemgetter(list_idxs_for_label)(np.array(self.texts)))
            batch_tweets = random.sample(full_tweets_list,min(len(full_tweets_list),self.batch_size_per_label))
            return batch_tweets, label_vec


#based on bert
class TextStyleClassification(nn.Module):
    def __init__(self, dropout=0.5,device=torch.device('cpu')):
        super(TextStyleClassification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        #for param in self.bert.parameters():
        #    param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        #self.linear1 = nn.Linear(768, 128)
        #self.linear2 = nn.Linear(128, NUM_OF_LABELS)
        self.linear1 = nn.Linear(768, NUM_OF_LABELS)
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax()


    def forward(self, input_id, mask):
        _, x = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False) # pooled_output is the embedding token of the [CLS] token for all batch
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        #x = self.relu(x)
        #x = self.linear2(x)
        #output = self.softmax(linear2_output)
        return x



def collate_fn(data): #for model based on bert
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


def train(model, optimizer, df_train, df_val, labels_set_dict, labels_idx_to_str, path_for_saving_last_model, path_for_saving_best_model, device, config, **kwargs):
    total_epochs = config['epochs']
    train_data_set = Dataset(df_train, labels_set_dict, config['inner_batch_size'],True)
    train_dataloader = torch.utils.data.DataLoader(train_data_set, collate_fn=collate_fn, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    val_data_set = Dataset(df_val, labels_set_dict, config['inner_batch_size'], True)
    val_dataloader = torch.utils.data.DataLoader(val_data_set, collate_fn=collate_fn,
                                                   batch_size=config['batch_size'], shuffle=True,
                                                   num_workers=config['num_workers'])

    print("Sanity check on train df...")
    criterion = nn.CrossEntropyLoss()

    print('Starting to train...')
    model = model.to(device)
    criterion = criterion.to(device)

    total_loss_train_list = []
    total_acc_train_list = []
    total_loss_val_list = []
    total_acc_val_list = []
    for epoch in range(config['epochs']):
        model.train()
        t000 = timeit.default_timer()
        total_loss_train = 0
        total_acc_train = 0
        total_acc_denum = 0
        for step, (tokenized_texts_list, labels, texts_list) in enumerate(pbar:= tqdm(train_dataloader, desc="Training", leave=False)): #model based on bert
            labels_cpu = labels
            labels = torch.from_numpy(np.asarray(labels)).to(device)
            #print(f"len of data: {labels.shape[0]}")
            outputs = model(tokenized_texts_list['input_ids'].to(device), tokenized_texts_list['attention_mask'].to(device)) #model based on bert
            outputs_cpu = outputs.cpu().data.numpy()
            if step==0:
                all_outputs = outputs_cpu
                all_labels_cpu = labels_cpu
            else:
                all_outputs = np.concatenate((all_outputs,outputs_cpu),axis=0)
                all_labels_cpu = np.concatenate((all_labels_cpu,labels_cpu),axis=0)

            loss = criterion(outputs, labels)
            total_loss_train += loss.item()

            #print(f"**************Predicted: {outputs.argmax(dim=1)}***************")
            #print(f"**************gt: {labels.argmax(dim=1)}***************")
            acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            #acc = (outputs_cpu.argmax(dim=1) == labels_cpu.argmax(dim=1)).sum().item()
            total_acc_train += acc
            total_acc_denum += outputs.shape[0]

            model.zero_grad()


            loss.backward()
            optimizer.step()


        total_loss_train_list.append(total_loss_train)
        total_acc_train_list.append(total_acc_train)

        outputs_mean = np.mean(all_outputs, axis=0)
        print(f"outputs_mean: {outputs_mean}")
        labels_mean = np.mean(all_labels_cpu, axis=0)
        print(f"labels_mean: {labels_mean}")
        total_acc_val = 0
        total_loss_val = 0
        model.eval()
        with torch.no_grad():
            for step, (tokenized_texts_list, val_labels, texts_list) in enumerate(
                    pbar := tqdm(val_dataloader, desc="Validation", leave=False, )):  # model based on bert
                val_labels = torch.from_numpy(np.asarray(val_labels)).to(device)
                outputs = model(tokenized_texts_list['input_ids'].to(device), tokenized_texts_list['attention_mask'].to(device)) #model based on bert

                loss = criterion(outputs, val_labels)
                total_loss_val += loss.item()

                acc = (outputs.argmax(dim=1) == val_labels.argmax(dim=1)).sum().item()
                total_acc_val += acc
        model.train()
        total_loss_val_list.append(total_loss_val)
        total_acc_val_list.append(total_acc_val)

        print(
            f'Epochs: {epoch + 1}/{total_epochs} | Train Loss: {total_loss_train / df_train.shape[0]: .3f} \
                    | Train Accuracy: {total_acc_train/total_acc_denum: .3f} \
                    | Val Loss: {total_loss_val / df_val.shape[0]: .3f} \
                    | Val Accuracy: {total_acc_val / df_val.shape[0]: .3f}')

        t001 = timeit.default_timer()
        print(f"time for single epoch is {(t001 - t000) / 60} min = {t001 - t000} sec.")
        log_dict = {'train/epoch': epoch,
                    'train/train_loss': total_loss_train / df_train.shape[0],
                    'train/train_acc': total_acc_train / df_train.shape[0],
                    'val/val_loss': total_loss_val / df_val.shape[0],
                    'val/val_acc': total_acc_val / df_val.shape[0]
                    }
        wandb.log({"log_dict": log_dict})
        if np.mod(epoch, 10) == 0:  # plot on every 10 epochs
            # save model
            print(f'Epoch = {epoch},Saving model to: {path_for_saving_last_model}...')
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        }, path_for_saving_last_model)  # finally check on all data training
    print('Finished to train.')


def getting_labels_map(df_train):
    labels_set_dict = {}
    labels_idx_to_str = {}
    for i, label in enumerate(set(df_train['label'])):
        labels_set_dict[label] = i
        labels_idx_to_str[i] = label
    print("labels_set_dict:")
    print(labels_set_dict)
    return labels_set_dict,labels_idx_to_str

def get_model_and_optimizer(config, path_for_loading_best_model, device):
    if config['load_model']: #load_model
        print(f"Loading model from: {path_for_loading_best_model}")
        model = TextStyleClassification(device=device)
        # optimizer = SGD(model.parameters(), lr=config['lr'])
        # take only non-frozen params:
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'],weight_decay=config['weight_decay'])
        checkpoint = torch.load(path_for_loading_best_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        #  train from scratch
        print("Train model from scratch")
        model = TextStyleClassification(device=device)
        # optimizer = SGD(model.parameters(), lr=config['lr'])
        # take only non-frozen params:
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    return model,optimizer


def get_train_val_data(data_set_path):
    ds = {}
    for type_set in data_set_path:
        ds[type_set] = {}
        for dataset_name in data_set_path[type_set]:
            with open(data_set_path[type_set][dataset_name], 'rb') as r:
                data = pickle.load(r)
            for k in data:
                ds[type_set][k] = {}
                #ds[type_set][k]['factual'] = data[k]['factual']  #todo: check if there is need to concatenate factual from senticap and flickrstyle10k
                #ds[type_set][k]['img_path'] = data[k]['image_path']
                if dataset_name == 'flickrstyle10k':
                    ds[type_set][k]['humor'] = data[k]['humor']
                    ds[type_set][k]['romantic'] = data[k]['romantic']
                elif dataset_name == 'senticap':
                    ds[type_set][k]['positive'] = data[k]['positive']
                    ds[type_set][k]['negative'] = data[k]['negative']
    return ds


def convert_ds_to_df(ds):
    for type_set in ds:
        all_data = {'label': [], 'text': []}
        for k in ds[type_set]:
            for style in ds[type_set][k]:
                if style == 'img_path':
                    continue
                all_data['label'].extend([style]*len(ds[type_set][k][style]))
                all_data['text'].extend(ds[type_set][k][style])
        #padd all lists to be in the same len
        #max_len = np.max([len(all_data['label']),len(all_data['text'])])
        #for l in all_data:
        #    all_data[l] += ['']*(max_len - len(all_data[l]))
        if type_set == 'train':
            df_train = pd.DataFrame(all_data)
        elif type_set == 'val':
            df_val = pd.DataFrame(all_data)

        pos_idxs = [i for i, x in enumerate(all_data['label']) if x == 'positive']
        neg_idxs = [i for i, x in enumerate(all_data['label']) if x == 'negative']
        humor_idxs = [i for i, x in enumerate(all_data['label']) if x == 'humor']
        romantic_idxs = [i for i, x in enumerate(all_data['label']) if x == 'romantic']
        factual_idxs = [i for i, x in enumerate(all_data['label']) if x == 'factual']
        print(f"{type_set}: len(pos_idxs)= {len(pos_idxs)}")
        print(f"{type_set}: len(neg_idxs)= {len(neg_idxs)}")
        print(f"{type_set}: len(humor_idxs)= {len(humor_idxs)}")
        print(f"{type_set}: len(romantic_idxs)= {len(romantic_idxs)}")
        print(f"{type_set}: len(factual_idxs)= {len(factual_idxs)}")
    return df_train, df_val

'''
def convert_ds_to_df(ds):
    for type_set in ds:
        style_data = {}
        for k in ds[type_set]:
            for style in ds[type_set][k]:
                if style == 'img_path':
                    continue
                if style not in style_data:
                    style_data[style] = list(ds[type_set][k][style])
                else:
                    style_data[style].extend(ds[type_set][k][style])
        #padd all lists to be in the same len
        max_len = 0
        for s in style_data:
            if len(style_data[s])>max_len:
                max_len = len(style_data[s])
        for s in style_data:
            style_data[s] += ['']*(max_len - len(style_data[s]))
        if type_set == 'train':
            df_train = pd.DataFrame(style_data)
        elif type_set == 'val':
            df_val = pd.DataFrame(style_data)
    return df_train, df_val
'''

def main():
    print(f"Cur dir:{os.getcwd()}")
    desired_cuda_num = "1"  #"2"  # "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = desired_cuda_num
    np.random.seed(112)  # todo there may be many more seeds to fix
    torch.cuda.manual_seed(112)

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

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{desired_cuda_num}" if use_cuda else "cpu")#todo: remove

    ###############################
    # text style classification
    data_dir = os.path.join(os.path.expanduser('~'), 'data')
    dataset_names = ['flickrstyle10k']  # ['senticap', 'flickrstyle10k']
    train_set_path = {}
    val_set_path = {}
    data_set_path = {'train': {}, 'val': {}}
    for dataset_name in dataset_names:
        data_set_path['train'][dataset_name] = os.path.join(data_dir, dataset_name, 'annotations', 'train.pkl')
        data_set_path['val'][dataset_name] = os.path.join(data_dir, dataset_name, 'annotations', 'val.pkl')

    ##################
    ds = get_train_val_data(data_set_path)
    df_train, df_val = convert_ds_to_df(ds)

    labels_set_dict, labels_idx_to_str = getting_labels_map(df_train)


    wandb.init(project='zero-shot-learning',
               config=config,
               resume=config['resume'],
               id=config['run_id'],
               mode=config['wandb_mode'],
               tags=config['tags'])

    model,optimizer = get_model_and_optimizer(config,path_for_loading_best_model,device)

    '''if config['plot_only_clustering']:
        log_dict_train = plot_graph_on_all_data(df_train,
                                                labels_set_dict, labels_idx_to_str, device, model,
                                                config['inner_batch_size'], config['batch_size'],
                                                "final embedding on trainning set for best model",
                                                tgt_file_vec_emb, True, True, config['num_workers'],
                                                config['desired_labels'])
        print('finished to plot clustering for the best model.')
        exit(0)
    '''
    # senity_check(df_train)


    ##################
    train(model, optimizer, df_train, df_val, labels_set_dict, labels_idx_to_str, path_for_saving_last_model,path_for_saving_best_model, device,config)
    # evaluate(model,  filtered_df_test, labels_set_dict, device, config,pos_combinations_labels,neg_combinations_labels)

    print('finish!')


if __name__ == '__main__':
    main()
