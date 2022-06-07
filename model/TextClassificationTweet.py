import pandas as pd
import os
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
import matplotlib

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, labels,tokenizer):
        self.labels = [labels[label] for label in df['User']]
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['Tweet']]

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

    def __init__(self, dropout=0.05):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 9)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False) # pooled_output is the embedding token of the [CLS] token for all batch
        #dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(pooled_output)
        #final_layer = self.relu(linear_output)
        return linear_output


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


def train(model, train_data, val_data, learning_rate, epochs, labels, tokenizer,batch_size):
    print('Starting to train...')
    train, val = Dataset(train_data, labels, tokenizer), Dataset(val_data, labels, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        #for step, (anchor_tweet, positive_tweet, negative_tweet, anchor_label) in enumerate(
        #        tqdm(train_dataloader, desc="Training", leave=False)):
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            '''
            anchor_label = anchor_label.to(device)
            anchor_mask = anchor_tweet['attention_mask'].to(device)
            anchor_input_id = anchor_tweet['input_ids'].squeeze(1).to(device)
            positive_mask = positive_tweet['attention_mask'].to(device)
            positive_input_id = positive_tweet['input_ids'].squeeze(1).to(device)
            negative_mask = negative_tweet['attention_mask'].to(device)
            negative_input_id = negative_tweet['input_ids'].squeeze(1).to(device)

            anchor_output = model(anchor_input_id, anchor_mask)
            positive_output = model(positive_input_id, positive_mask)
            negative_output = model(negative_input_id, negative_mask)

            batch_loss = criterion(anchor_output, positive_output, negative_output)
            '''
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label)
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

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data,labels, tokenizer,batch_size):
    print('Starting to evaluate...')
    test = Dataset(test_data,labels, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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

    print(f' Accuracy: {total_acc_test / len(test_data): .3f}')



def main():
    print('Start!')
    #base_path = os.getcwd()
    base_path = '~/zero-shot-style/'
    print('base_path = '+base_path)

    datapath = 'preprocessed_data.csv'
    df = pd.read_csv(base_path + datapath)
    df.head()

    df.groupby(['User']).size().plot.bar()

    # df.groupby(['User']).size().plot.bar()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    labels = {}
    for i, label in enumerate(set(df.iloc[:, 0])):
        labels[label] = i
    # labels = {'business':0,
    #           'entertainment':1,
    #           'sport':2,
    #           'tech':3,
    #           'politics':4
    #           }

    np.random.seed(112)
    print('Splitting DB to train, val and test data frames.')
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    #print(len(df_train), len(df_val), len(df_test))

    EPOCHS = 8
    model = BertClassifier()
    LR = 1e-4

    batch_size = 16

    train(model, df_train, df_val, LR, EPOCHS, labels,tokenizer,batch_size)

    evaluate(model, df_test,labels, tokenizer,batch_size)
    print('  finish!')


if __name__=='__main__':
    main()