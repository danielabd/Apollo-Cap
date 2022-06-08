import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
import matplotlib
##matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#fig = plt.figure()
#fig.show()
import random



class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, labels,tokenizer):
        self.labels = [labels[label] for label in df['User']]
        self.labels_set = set(self.labels)
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['Tweet']]
        self.index = df.index.values
        self.batch_size_per_label = 10
        pass

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)
        #return len(self.labels_set)

    '''
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    '''

    def __getitem__(self, item):
        '''
        label = self.labels_set[item]
        list_idxs_for_label = self.index[np.array(self.labels) == label]
        relevant_idxs = np.random.random_integers(0, len(list_idxs_for_label), size=(1, self.batch_size_per_label))[0]
        data = {}
        for i in relevant_idxs:
            data[label] = self.texts[i]
        '''
        
        anchor_text = self.texts[item]
        anchor_label = self.labels[item]

        positive_list = self.index[self.index != item][np.array(self.labels)[self.index != item] == anchor_label]
        positive_item = random.choice(positive_list)
        #positive_text = np.array(self.texts)[self.index==positive_item]
        positive_text = self.index[np.where(self.index == positive_item)]

        negative_list = self.index[self.index != item][np.array(self.labels)[self.index != item] != anchor_label]
        negative_item = random.choice(negative_list)
        negative_text = self.index[np.where(self.index == negative_item)]

        return anchor_text, positive_text, negative_text, anchor_label



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



def train(model, train_data, learning_rate, epochs, labels,  tokenizer,batch_size):
    print('Starting to train...')
    # train, val = Dataset(train_data, labels, tokenizer), Dataset(val_data, labels, tokenizer)
    # train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    # val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    train = Dataset(train_data, labels, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #criterion = nn.CrossEntropyLoss()
    criterion = torch.jit.script(TripletLoss())
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    model.train()
    for epoch in range(epochs):
        running_loss = []
        #total_acc_train = 0
        #total_loss_train = 0

        for step, (anchor_text, positive_text, negative_text, anchor_label) in enumerate(
                tqdm(train_dataloader, desc="Training", leave=False)):

            anchor_label = anchor_label.to(device)
            anchor_mask = anchor_text['attention_mask'].to(device)
            anchor_input_id = anchor_text['input_ids'].squeeze(1).to(device)
            positive_mask = positive_text['attention_mask'].to(device)
            positive_input_id = positive_text['input_ids'].squeeze(1).to(device)
            negative_mask = negative_text['attention_mask'].to(device)
            negative_input_id = negative_text['input_ids'].squeeze(1).to(device)

            anchor_output = model(anchor_input_id, anchor_mask)
            positive_output = model(positive_input_id, positive_mask)
            negative_output = model(negative_input_id, negative_mask)

            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))

        # save model
        torch.save({"model_state_dict": model.state_dict(),
                    "optimzier_state_dict": optimizer.state_dict()
                    }, "trained_model.pth")



def evaluate(model,val_data, labels, tokenizer,batch_size):
    val = Dataset(val_data, labels, tokenizer)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_results = []
    labels = []
    model.eval()
    with torch.no_grad():
        for text, _, _, label in tqdm(val_dataloader):
            #anchor_label = anchor_label.to(device)
            text_mask = text['attention_mask'].to(device)
            text_input_id = text['input_ids'].squeeze(1).to(device)

            train_results.append(model(text_input_id, text_mask).cpu().numpy())
            #train_results.append(model(text.to(device)).cpu().numpy())
            labels.append(label)

    train_results = np.concatenate(train_results)
    labels = np.concatenate(labels)
    train_results.shape

    #Visualization

    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = train_results[labels == label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

    plt.legend()
    plt.show()


'''for classification
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
'''


def main():
    print('Start!')
    #base_path = os.getcwd()
    base_path = '~/zero-shot-style/'
    print('base_path = '+base_path)

    datapath = 'preprocessed_data.csv'
    df = pd.read_csv(base_path + datapath)
    df.head()

    df.groupby(['User']).size().plot.bar()
    '''
    #analyze the data
    tweet_user_dict = {}
    for i, user in enumerate(df['User']):
        tweet = df['Tweet'][i]
        num_of_words = len(tweet.split())
        if user not in tweet_user_dict:
            tweet_user_dict[user] = {}
            tweet_user_dict[user]['tweets'] = [tweet]
            tweet_user_dict[user]['num_of_words'] = [num_of_words]
            tweet_user_dict[user]['max_len'] = num_of_words
            tweet_user_dict[user]['min_len'] = num_of_words
        else:
            tweet_user_dict[user]['tweets'].append(tweet)
            tweet_user_dict[user]['num_of_words'].append(num_of_words)
            if num_of_words>tweet_user_dict[user]['max_len']:
                tweet_user_dict[user]['max_len']=num_of_words
            if num_of_words<tweet_user_dict[user]['min_len']:
                tweet_user_dict[user]['min_len'] = num_of_words
    for user in tweet_user_dict:
        a=plt.hist(tweet_user_dict[user]['num_of_words'])
        pass
    '''

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

    EPOCHS = 1#8
    model = BertClassifier()
    LR = 1e-4

    batch_size = 8# 16-not enuff space in memory

    #train model
    train(model, df_train, LR, EPOCHS, labels, tokenizer,batch_size)



    evaluate(model, df_test, labels, tokenizer,batch_size)
    print('  finish!')


if __name__=='__main__':
    main()