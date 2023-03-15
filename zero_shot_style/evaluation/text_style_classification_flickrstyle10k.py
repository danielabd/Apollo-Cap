import operator
import os.path
import pickle
import random
from datetime import datetime

import wandb
from sklearn.metrics import f1_score
import pandas as pd
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel

from torch.optim import Adam
from tqdm import tqdm
import torch
import numpy as np
from transformers import BertTokenizer, BertConfig
import torch
from torchmetrics import Precision, Recall
from zero_shot_style.utils import parser, get_hparams

NUM_OF_CLASSES = 2  # 4
BERT_NUM_OF_LAYERS = 12

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def get_args():
    parser.add_argument('--config_file', type=str,
                        default=os.path.join('..', 'configs', 'flickrstyle10k_text_style_classification.yaml'),
                        help='full path to config file')
    parser.add_argument('--hidden_state_to_take', type=int, default=-2, help='hidden state of BERT totake')
    parser.add_argument('--last_layer_idx_to_freeze', type=int, default=-1, help='last_layer idx of BERT to freeze')
    parser.add_argument('--freeze_after_n_epochs', type=int, default=0, help='freeze BERT after_n_epochs')
    parser.add_argument('--scale_noise', type=float, default=0.0, help='scale of gaussian noise to add to the embedding vector of sentence')
    args = parser.parse_args()
    return args


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


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.05, device=torch.device('cpu'), hidden_state_to_take=-1, last_layer_idx_to_freeze=BERT_NUM_OF_LAYERS,
                 scale_noise=0):
        super(BertClassifier, self).__init__()
        bert_config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-cased', config=bert_config)
        self.freeze_layers(last_layer_idx_to_freeze)
        self.hidden_state_to_take = hidden_state_to_take
        self.scale_noise = scale_noise

        #for param in self.bert.parameters():
        #   param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        #self.linear = nn.Linear(768, NUM_OF_CLASSES)
        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.scale_noise = scale_noise

    def forward(self, input_id, mask):
        '''
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
        '''
        outputs = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        hidden_states = outputs[2]
        # embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]
        x = attention_hidden_states[self.hidden_state_to_take][:, 0, :]  # CLS output of self.hidden_state_to_take layer.
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)

        # #add gaussian noise
        x = x + torch.randn_like(x) * self.scale_noise
        x = x / x.norm(dim=-1, keepdim=True)

        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

    def freeze_layers(self, last_layer_idx_to_freeze=BERT_NUM_OF_LAYERS):
        '''

        #:param freeze_layers: list of layers num to freeze
        :param last_layer_idx_to_freeze: int , from which layer need to freeze the model
        :return:
        '''
        for layer_idx in range(BERT_NUM_OF_LAYERS):
            for param in list(self.bert.encoder.layer[layer_idx].parameters()):
                if layer_idx <= last_layer_idx_to_freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                # print(f"BERT layers up to {layer_idx} are frozen.")


    def set_noise(self, scale_noise):
        self.scale_noise = 0


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

def train(model, optimizer, df_train, df_val, labels_set_dict, labels_idx_to_str, path_for_saving_last_model,
          path_for_saving_best_model, device, config):
    print("Training the model...")
    train_data_set, val_data_set = Dataset(df_train, labels_set_dict), Dataset(df_val, labels_set_dict)
    train_dataloader = torch.utils.data.DataLoader(train_data_set, collate_fn=collate_fn,
                                                   batch_size=config['batch_size'], shuffle=True,
                                                   num_workers=config['num_workers'])
    val_dataloader = torch.utils.data.DataLoader(val_data_set, collate_fn=collate_fn, batch_size=config['batch_size'],
                                                 shuffle=True, num_workers=config['num_workers'])

    print('Starting to train...')
    model = model.to(device)

    criterion = nn.BCELoss()

    best_f1_score_val = 0
    for epoch in range(config['epochs']):
        model.train()
        model.set_noise(config['scale_noise'])
        if epoch == config['freeze_after_n_epochs']:
            model.freeze_layers(BERT_NUM_OF_LAYERS)

        total_acc_train = 0
        total_loss_train = 0

        train_preds = []
        train_targets = []
        for step, (tokenized_texts_list, labels, texts_list) in enumerate(
                pbar := tqdm(train_dataloader, desc="Training", leave=False, )):  # model based on bert
            train_targets.extend(labels)
            train_label = torch.from_numpy(np.asarray(labels)).to(device)
            outputs = model(tokenized_texts_list['input_ids'].to(device),
                            tokenized_texts_list['attention_mask'].to(device))  # model based on bert
            train_preds.extend([i[0] for i in outputs.cpu().data.numpy()])

            train_label2 = torch.from_numpy(np.asarray([[float(i)] for i in labels])).to(device).float()
            outputs = outputs.float()
            batch_loss = criterion(outputs, train_label2)

            total_loss_train += batch_loss.item()

            outputs_bin = torch.round(torch.tensor([out[0] for out in outputs])).to(device)
            acc = (outputs_bin == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # train_preds_t = torch.tensor(train_preds)
        # train_preds_bin = torch.round(torch.tensor(train_preds))
        # train_targets_t = torch.tensor(train_targets)
        # precision_i = Precision(average='weighted', task='binary',  num_classes=len(set(np.array(train_targets_t))),multiclass=True)
        # precision = precision_i(train_preds_bin, train_targets_t)
        # recall_i = Recall(average='weighted', num_classes=len(set(np.array(train_targets_t))), task='multiclass', multiclass=True)
        # recall = recall_i(train_preds_bin, train_targets_t)

        # precision = Precision(preds, targets)
        # recall = Recall(preds, targets)

        # f1_score_train = 2 * (precision * recall) / (precision + recall)

        #remove save last model bacause size
        # if np.mod(epoch,10) == 0:
        #     print(f'Saving model to: {path_for_saving_last_model}...')
        #     torch.save({"model_state_dict": model.state_dict(),
        #                 "optimizer_state_dict": optimizer.state_dict(),
        #                 }, path_for_saving_last_model)
        total_acc_val = 0
        total_loss_val = 0
        print("Calculate  validation...")
        val_preds = []
        val_targets = []
        model.set_noise(0)
        model.eval()
        with torch.no_grad():
            for step, (tokenized_texts_list, val_labels, texts_list) in enumerate(
                    pbar := tqdm(val_dataloader, desc="Validation", leave=False, )):
                val_targets.extend(val_labels)
                val_labels = torch.from_numpy(np.asarray(val_labels)).to(device)
                outputs = model(tokenized_texts_list['input_ids'].to(device),
                                tokenized_texts_list['attention_mask'].to(device))  # model based on bert
                val_preds.extend([i[0] for i in outputs.cpu().data.numpy()])
                val_label2 = torch.from_numpy(np.asarray([[float(i)] for i in val_labels])).to(device).float()
                outputs = outputs.float()
                batch_loss = criterion(outputs, val_label2)
                total_loss_val += batch_loss.item()

                outputs_bin = torch.round(torch.tensor([out[0] for out in outputs])).to(device)
                acc = (outputs_bin == val_labels).sum().item()
                total_acc_val += acc

            # train_preds_t = torch.tensor(train_preds)
            val_preds_bin = torch.round(torch.tensor(val_preds))
            val_targets_t = torch.tensor(val_targets)
            precision_i = Precision(average='weighted', task='binary', num_classes=len(set(np.array(val_targets_t))),
                                    multiclass=True)
            precision = precision_i(val_preds_bin, val_targets_t)
            recall_i = Recall(average='weighted', num_classes=len(set(np.array(val_targets_t))), task='multiclass', multiclass=True)
            recall = recall_i(val_preds_bin, val_targets_t)

            f1_score_val = 2*(precision*recall)/(precision+recall)
            #f1_score_val = f1_score(label_cpu, output_cpu, average='weighted')


            print(f"f1_score_val:{f1_score_val}")

        if f1_score_val>best_f1_score_val:
            print(f'f1_score_val = {f1_score_val} improved.\nSaving ***best**** model to: {path_for_saving_best_model}...')
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        }, path_for_saving_best_model)
            best_f1_score_val = f1_score_val


        print(
            f'Epochs: {epoch + 1} \
                | Train Loss: {total_loss_train / len(df_train): .3f} \
                | Train Accuracy: {total_acc_train / len(df_train): .3f} \
                | Val Loss: {total_loss_val / len(df_val): .3f} \
                | Val Accuracy: {total_acc_val/len(df_val): .3f} \
                | f1_score_val: {f1_score_val: .3f} \
                | best_f1_score_val: {best_f1_score_val: .3f}')
        log_dict = {'train/epoch': epoch,
                    'train/loss_train': total_loss_train / len(df_train),
                    'train/acc_train': total_acc_train / len(df_train),
                    'val/loss_val': total_loss_val / len(df_val),
                    'val/acc_val': total_acc_val/len(df_val ),
                    'val/f1_score_val': f1_score_val,
                    'val/best_f1_score_val': best_f1_score_val}
        wandb.log(log_dict)
    print("finish train")


def evaluate(model, all_df, labels_set_dict, device, config):
    evaluation = {}
    for set_name in all_df:
        print(f"Evaluating the model on {set_name} set:")
        evaluation[set_name] = {}
        df = all_df[set_name]
        test_data_set = Dataset(df, labels_set_dict)
        test_dataloader = torch.utils.data.DataLoader(test_data_set, collate_fn=collate_fn,
                                                       batch_size=config['batch_size'], shuffle=True,
                                                       num_workers=config['num_workers'])

        model = model.to(device)
        model.freeze_layers(BERT_NUM_OF_LAYERS)
        model.set_noise(0)

        model.eval()
        total_acc_test = 0
        total_loss_test = 0
        test_preds = []
        test_targets = []
        criterion = nn.BCELoss()
        with torch.no_grad():
            for step, (tokenized_texts_list, labels, texts_list) in enumerate(
                    pbar := tqdm(test_dataloader, desc=f"eval {set_name}", leave=False, )):  # model based on bert
                test_targets.extend(labels)
                test_label = torch.from_numpy(np.asarray(labels)).to(device)
                outputs = model(tokenized_texts_list['input_ids'].to(device),
                                tokenized_texts_list['attention_mask'].to(device))  # model based on bert
                test_preds.extend([i[0] for i in outputs.cpu().data.numpy()])

                test_label2 = torch.from_numpy(np.asarray([[float(i)] for i in labels])).to(device).float()
                outputs = outputs.float()
                batch_loss = criterion(outputs, test_label2)
                total_loss_test += batch_loss.item()

                outputs_bin = torch.round(torch.tensor([out[0] for out in outputs])).to(device)
                acc = (outputs_bin == test_label).sum().item()
                total_acc_test += acc


        print(f"test loss: {total_loss_test / len(df): .3f}")

        test_preds_bin = torch.round(torch.tensor(test_preds))
        test_targets_t = torch.tensor(test_targets)
        precision_i = Precision(average='weighted', task='binary', num_classes=len(set(np.array(test_targets_t))),
                                multiclass=True)
        precision = precision_i(test_preds_bin, test_targets_t)
        recall_i = Recall(average='weighted', num_classes=len(set(np.array(test_targets_t))), task='multiclass',
                          multiclass=True)
        recall = recall_i(test_preds_bin, test_targets_t)

        f1_score_test = 2 * (precision * recall) / (precision + recall)
        # f1_score_val = f1_score(label_cpu, output_cpu, average='weighted')

        print(f"f1_score_val:{f1_score_test}")

        total_acc_test_for_all_data = total_acc_test / len(df)

        print(f'Test Accuracy: {total_acc_test_for_all_data: .3f}')
        evaluation[set_name]['f1_score'] = f1_score_test
        evaluation[set_name]['acc'] = total_acc_test_for_all_data
        evaluation[set_name]['loss'] = total_loss_test / len(df)
        print(f"finish to evaluate {set_name}")
    for set_name in evaluation:
        print(f"evaluation of {set_name}:")
        for metric in evaluation[set_name]:
            print(f"{set_name}_{metric} = {evaluation[set_name][metric]}")
    print("Finish!")

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
        # ######## #clear data from ' .' in the end of captions
        # for k in data:
        #     for style in data[k]:
        #         for i,sen in enumerate(data[k][style]):
        #             if data[k][style][i][-2:]==' .':
        #                 data[k][style][i] = data[k][style][i][:-2]
        # with open(data_set_path[data_type], 'wb') as file:
        #     pickle.dump(data, file)
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

def get_model_and_optimizer(config, path_for_loading_best_model, device):
    if config['load_model'] or config['task']=='test':  # load_model
        print(f"Loading model from: {path_for_loading_best_model}")
        model = BertClassifier(device=device, hidden_state_to_take=config['hidden_state_to_take'],
                               last_layer_idx_to_freeze=config['last_layer_idx_to_freeze'], scale_noise=config['scale_noise'])
        model.to(device)
        optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        if 'cuda' in device.type:
            checkpoint = torch.load(path_for_loading_best_model, map_location='cuda:0')
        else:
            checkpoint = torch.load(path_for_loading_best_model, map_location=torch.device(device.type))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        #  train from scratch
        print("Train model from scratch")
        model = BertClassifier(device=device, hidden_state_to_take=config['hidden_state_to_take'],
                               last_layer_idx_to_freeze=config['last_layer_idx_to_freeze'], scale_noise=config['scale_noise'])
        optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    return model, optimizer



def main():
    args = get_args()
    config = get_hparams(args)
    print(f"config_file = {config['config_file']}")

    cur_date = datetime.now().strftime("%d_%m_%Y")
    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print(f"cur time is: {cur_time}")

    wandb.init(project='text-style-classification',
               config=config,
               resume=config['resume'],
               id=config['run_id'],
               mode=config['wandb_mode'],  # disabled, offline, online'
               tags=config['tags'])

    config['training_name'] = f'{wandb.run.id}-{wandb.run.name}'
    print(f"training_name = {config['training_name']}")
    desired_cuda_num = 0

    np.random.seed(112)  # todo there may be many more seeds to fix
    torch.cuda.manual_seed(112)
    # overwrite_pairs = True  # todo

    checkpoints_dir = os.path.join(os.path.expanduser('~'), 'checkpoints')
    global_dir_name_for_save_models = os.path.join(checkpoints_dir, config['global_dir_name_for_save_models'])
    if not os.path.isdir(global_dir_name_for_save_models):
        os.makedirs(global_dir_name_for_save_models)
    experiment_dir_date = os.path.join(checkpoints_dir,config['global_dir_name_for_save_models'], cur_date)
    if not os.path.isdir(experiment_dir_date):
        os.makedirs(experiment_dir_date)
    experiment_dir = os.path.join(checkpoints_dir,config['global_dir_name_for_save_models'], cur_date, cur_time)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    data_dir = os.path.join(os.path.expanduser('~'), 'data')

    data_set_path = {'train': {}, 'val': {}, 'test': {}}
    for data_type in ['train', 'val', 'test']:
        data_set_path[data_type] = os.path.join(data_dir, config['data_name'], 'annotations',
                                                             data_type + '.pkl')

    path_for_saving_last_model = os.path.join(experiment_dir, config['model_name'])
    path_for_saving_best_model = os.path.join(experiment_dir, config['best_model_name'])
    # path_for_loading_best_model = os.path.join(checkpoints_dir, 'best_model',dataset_names[0], config['best_model_name'])

    if 'path_for_loading_best_model' in config and config['path_for_loading_best_model']:
        path_for_loading_best_model = config['path_for_loading_best_model']
    else:
        path_for_loading_best_model = os.path.join(checkpoints_dir, 'best_models', config['data_name'], config['best_model_name'])


    use_cuda = torch.cuda.is_available()
    # device = torch.device(f"cuda:{config['desired_cuda_num']}" if use_cuda else "cpu")  # todo: remove
    device = torch.device("cuda" if use_cuda else "cpu")  # todo: remove



    ds = get_train_val_data(data_set_path)
    df_train, df_val, df_test = convert_ds_to_df(ds, data_dir)
    # #######
    # #todo: debug
    # df_train = df_train.iloc[:3,:]
    # df_val = df_val.iloc[:3,:]
    # ######
    print(len(df_train), len(df_val), len(df_test))
    print(f"labels: {config['labels_set_dict']}")

    # df_train,df_test = get_train_test_data(config, config['undesired_label'])
    model, optimizer = get_model_and_optimizer(config, path_for_loading_best_model, device)
    # config['labels_set_dict'],config['labels_idx_to_str'] = getting_labels_map(df_train)

    if config['task'] == 'train':
        train(model, optimizer, df_train, df_val, config['labels_set_dict'], config['labels_idx_to_str'], path_for_saving_last_model,
              path_for_saving_best_model, device,  config)
    elif config['task'] == 'test':
        all_df = {}
        for d in config['test_on_df']:
            if d=='train':
                all_df[d] = df_train
            elif d=='val':
                all_df[d] = df_val
            elif d=='test':
                all_df[d] = df_test
        evaluate(model, all_df, config['labels_set_dict'], device, config)

    print("finish main")


if __name__=='__main__':
    main()
