import os.path
import pickle
from sklearn.metrics import f1_score

import pandas as pd
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel

from torch.optim import Adam
from tqdm import tqdm
import torch
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
'''
labels = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }
'''

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, labels):

        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs, labels, batch_size, desired_cuda_num):
    print("Trainin the model...")
    train, val = Dataset(train_data, labels), Dataset(val_data, labels)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{desired_cuda_num}" if use_cuda else "cpu")  # todo: remove

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

            label_cpu = train_label.cpu().data.numpy()
            output_cpu = output.argmax(dim=1).cpu().data.numpy()
            f1_score_val = f1_score(label_cpu, output_cpu, average='weighted')
            print(f"f1_score_val:{f1_score_val}")

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

    path_for_saving_last_model = os.path.join(os.path.expanduser('~'),'checkpoints','best_model','best_text_style_classification_model.pth')
    print(f'Saving model to: {path_for_saving_last_model}...')
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                }, path_for_saving_last_model)
    print("finish train")

def evaluate(model, test_data, labels, desired_cuda_num):
    test = Dataset(test_data, labels)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=16)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{desired_cuda_num}" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    total_acc_test_for_all_data = total_acc_test / len(test_data)

    print(f'Test Accuracy: {total_acc_test_for_all_data: .3f}')
    print("finish evaluate")
    return total_acc_test_for_all_data



def get_train_val_data(data_set_path):
    ds = {}
    for set_type in data_set_path:  # ['train', 'val', 'test']
        ds[set_type] = {}
        for dataset_name in data_set_path[set_type]:
            with open(data_set_path[set_type][dataset_name], 'rb') as r:
                data = pickle.load(r)
            for k in data:
                ds[set_type][k] = {}
                #ds[set_type][k]['factual'] = data[k]['factual']  #todo: check if there is need to concatenate factual from senticap and flickrstyle10k
                #ds[set_type][k]['img_path'] = data[k]['image_path']
                if dataset_name == 'flickrstyle10k':
                    ds[set_type][k]['humor'] = data[k]['humor']
                    ds[set_type][k]['romantic'] = data[k]['romantic']
                elif dataset_name == 'senticap':
                    ds[set_type][k]['positive'] = data[k]['positive']
                    ds[set_type][k]['negative'] = data[k]['negative']
    return ds


def convert_ds_to_df(ds,data_dir):
    df_train=None; df_val = None; df_test=None
    for set_type in ds:  # ['train', 'val', 'test']
        all_data = {'category': [], 'text': []}
        for k in ds[set_type]:
            for style in ds[set_type][k]:
                if style == 'img_path':
                    continue
                all_data['category'].extend([style]*len(ds[set_type][k][style]))
                all_data['text'].extend(ds[set_type][k][style])
        # padd all lists to be in the same len
        # max_len = np.max([len(all_data['category']),len(all_data['text'])])
        # for l in all_data:
        #    all_data[l] += ['']*(max_len - len(all_data[l]))

        if set_type == 'train':
            df_train = pd.DataFrame(all_data)
            df_train.to_csv(os.path.join(data_dir,'train.csv'))
        elif set_type == 'val':
            df_val = pd.DataFrame(all_data)
            df_val.to_csv(os.path.join(data_dir, 'val.csv'))
        elif set_type == 'test':
            df_test = pd.DataFrame(all_data)
            df_test.to_csv(os.path.join(data_dir, 'test.csv'))
        '''
        pos_idxs = [i for i, x in enumerate(all_data['category']) if x == 'positive']
        neg_idxs = [i for i, x in enumerate(all_data['category']) if x == 'negative']
        humor_idxs = [i for i, x in enumerate(all_data['category']) if x == 'humor']
        romantic_idxs = [i for i, x in enumerate(all_data['category']) if x == 'romantic']
        factual_idxs = [i for i, x in enumerate(all_data['category']) if x == 'factual']
        print(f"{set_type}: len(pos_idxs)= {len(pos_idxs)}")
        print(f"{set_type}: len(neg_idxs)= {len(neg_idxs)}")
        print(f"{set_type}: len(humor_idxs)= {len(humor_idxs)}")
        print(f"{set_type}: len(romantic_idxs)= {len(romantic_idxs)}")
        print(f"{set_type}: len(factual_idxs)= {len(factual_idxs)}")
        '''
    return df_train, df_val, df_test

def main():
    desired_cuda_num = 0

    batch_size = 16  # 2
    data_dir = os.path.join(os.path.expanduser('~'), 'data')
    dataset_names = ['senticap', 'flickrstyle10k']
    path_to_csv_file = os.path.join(data_dir,'_'.join(dataset_names)+'.csv')
    data_set_path = {'train': {}, 'val': {}, 'test': {}}
    for dataset_name in dataset_names:
        for set_type in ['train', 'val', 'test']:
            data_set_path[set_type][dataset_name] = os.path.join(data_dir, dataset_name, 'annotations', set_type+'.pkl')

    ds = get_train_val_data(data_set_path)
    df_train, df_val, df_test = convert_ds_to_df(ds, data_dir)


    datapath = os.path.join(os.path.expanduser('~'),'data','bbc-text.csv')
    #datapath = path_to_csv_file
    df = pd.read_csv(datapath)
    df.head()
    df.groupby(['category']).size().plot.bar()
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
   
    example_text = 'I will watch Memento tonight'
    bert_input = tokenizer(example_text, padding='max_length', max_length=10,
                           truncation=True, return_tensors="pt")

    print(bert_input['input_ids'])
    print(bert_input['token_type_ids'])
    print(bert_input['attention_mask'])

    example_text = tokenizer.decode(bert_input.input_ids[0])

    print(example_text)
    
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    '''
    print(len(df_train), len(df_val), len(df_test))

    labels = {}
    for i,label in enumerate(list(set(list(df_train['category'])+list(df_val['category'])+list(df_test['category'])))):
        labels[label] = i
    print(f"labels: {labels}")
    EPOCHS = 20
    model = BertClassifier()
    LR = 1e-6

    train(model, df_train, df_val, LR, EPOCHS, labels, batch_size, desired_cuda_num)

    total_acc_test_for_all_data = evaluate(model, df_test, labels, desired_cuda_num)


    print("finish main")


if __name__=='__main__':
    main()
