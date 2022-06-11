import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam, SGD
from tqdm import tqdm
import matplotlib
##matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#fig = plt.figure()
#fig.show()
import random
import operator
import random
import Mining
from Mining import *
import sklearn
from sklearn.manifold import TSNE
from Plot import scatter
from argparse import ArgumentParser
import wandb

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, labels,tokenizer,inner_batch_size):
        self.labels = [labels[label] for label in df['User']]
        self.labels_set = list(set(self.labels))
        self.texts = [text for text in df['Tweet']]
        self.tokenizer = tokenizer
        #self.texts = [tokenizer(text,
        #                       padding='max_length', max_length = 512, truncation=True,
        #                        return_tensors="pt") for text in df['Tweet']]
        self.index = df.index.values
        self.batch_size_per_label = inner_batch_size
        pass

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels_set)

    def __getitem__(self, item):
        label = self.labels_set[item]
        #list_idxs_for_label = self.index[np.array(self.labels) == label]
        list_idxs_for_label = [np.array(self.labels) == label]
        full_tweets_list = list(operator.itemgetter(list_idxs_for_label)(np.array(self.texts)))
        batch_tweets = random.sample(full_tweets_list,min(len(full_tweets_list),self.batch_size_per_label))
        #tweets_list = list(operator.itemgetter(list_idxs_for_label)(np.array(self.texts)))
        #batch_per_label = self.tokenizer(data_text, padding='max_length', max_length=512, truncation=True,
        #                                 return_tensors="pt")
        return batch_tweets, label



class BertClassifier(nn.Module):

    def __init__(self, dropout=0.05):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad_(True)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(700, 600)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False) # pooled_output is the embedding token of the [CLS] token for all batch
        #dropout_output = self.dropout(pooled_output)
        relu_output = self.relu(pooled_output)
        linear1_output = self.linear1(relu_output)
        #relu_output = self.relu(linear1_output)
        #linear2_output = self.linear2(relu_output)
        output = torch.nn.functional.normalize(linear1_output)

        #output = torch.nn.functional.normalize(pooled_output)
        return output



def collate_fn(data):
    tweets_list = []
    labels_list = []
    for list_for_label in data:
        for tweet in list_for_label[0]:
            tweets_list.append(tweet)
            labels_list.append(list_for_label[1])
    #tweets_list = [tweet for tweet in list_for_label for list_for_label in data]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_tweets_list = tokenizer(tweets_list, padding=True,#padding='max_length', max_length=512, truncation=True,
                                     return_tensors="pt")
    return tokenized_tweets_list, labels_list


def plot_graph_on_all_data(train_data, labels, tokenizer, device, fig_path, model,inner_batch_size, state, batch_size):
    ########plot initial graph#########
    tsne = TSNE(random_state=0)
    train = Dataset(train_data, labels, tokenizer,inner_batch_size)
    train_dataloader = torch.utils.data.DataLoader(train, collate_fn=collate_fn, batch_size = batch_size, shuffle=True,
                                                   num_workers=0)
    x_train, y_train = next(iter(train_dataloader))
    x_train = x_train.to(device)
    y_train = torch.from_numpy(np.asarray(y_train)).to(device)
    y_train = y_train.to(device)

    if state=='final':
        outputs = model(x_train['input_ids'], x_train['attention_mask'])
        train_tsne_embeds = tsne.fit_transform(outputs.flatten(1).cpu().detach().numpy())
    elif state=='initial':
        train_tsne_embeds = tsne.fit_transform(x_train['input_ids'].flatten(1).cpu().detach().numpy())

    scatter(train_tsne_embeds, y_train.cpu().numpy(),fig_path, subtitle=f'online hard Original MNIST distribution (train set)')
    #scatter(test_tsne_embeds, y_test.cpu().numpy(), subtitle=f'online hard Original MNIST distribution (test set)')
    ######## end of plot initial graph #########


def train(model, source_train_data,val_data, learning_rate, epochs, labels,  tokenizer,batch_size,margin,inner_batch_size):
    print('Starting to train...')
    # train, val = Dataset(train_data, labels, tokenizer), Dataset(val_data, labels, tokenizer)
    # train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    # val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    source_labels = labels
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    fig_path_train = '/home/bdaniela/zero-shot-style/initial_scatter_train.png'
    plot_graph_on_all_data(source_train_data, labels, tokenizer, device, fig_path_train, model,inner_batch_size, state='initial',
                           batch_size=len(set(source_train_data['User'])))

    fig_path_val = '/home/bdaniela/zero-shot-style/initial_scatter_val.png'
    plot_graph_on_all_data(val_data, labels, tokenizer, device, fig_path_val, model,inner_batch_size, state='initial',
                           batch_size=len(set(val_data['User'])))

    train_data = Dataset(source_train_data, labels, tokenizer,inner_batch_size)
    train_dataloader = torch.utils.data.DataLoader(train_data, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)

    #criterion = nn.CrossEntropyLoss()
    #criterion = torch.jit.script(TripletLoss())
    optimizer = SGD(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.to(device)
        #criterion = criterion.to(device)

    model.train()
    for epoch in range(epochs):
        running_loss = []
        #total_acc_train = 0
        #total_loss_train = 0
        for step, (tokenized_tweets_list, labels) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            labels = torch.from_numpy(np.asarray(labels)).to(device)
            masks = tokenized_tweets_list['attention_mask'].to(device)
            input_ids = tokenized_tweets_list['input_ids'].squeeze(1).to(device)
            outputs = model(input_ids, masks)

            #using tripletloss

            loss, num_positive_triplets, num_valid_triplets, all_triplet_loss_avg = Mining.online_mine_all(labels, outputs,margin,device=device)
            fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
            #loss, pos_mask, neg_mask = Mining.online_mine_hard(labels, outputs, margin=margin, squared=True,
            #                                                   device=device)


            #loss = criterion(output,labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
            print("\nEpoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)),'\n')
            log_dict = {'train/epoch': epoch,
                        'train/train_loss': loss.cpu().detach().numpy(),
                        'train/fraction_positive_triplets': fraction_positive_triplets,
                        'train/num_positive_triplets': num_positive_triplets,
                        'train/all_triplet_loss_avg': all_triplet_loss_avg}
            #fig_path_val = '/home/bdaniela/zero-shot-style/'+str(step)+'_scatter_val.png'
            #plot_graph_on_all_data(val_data, source_labels, tokenizer, device, fig_path_val, model,inner_batch_size, state='initial',
            #                       batch_size=len(set(val_data['User'])))
            wandb.log(log_dict)


        # save model
        torch.save({"model_state_dict": model.state_dict(),
                    "optimzier_state_dict": optimizer.state_dict()
                    }, "trained_model.pth")
    print('Finished to train')
    #finally check on all data training

    fig_path = '/home/bdaniela/zero-shot-style/final_scatter_train.png'
    plot_graph_on_all_data(source_train_data, source_labels, tokenizer, device, fig_path, model, inner_batch_size, state='final',batch_size=len(set(source_train_data['User'])))

    fig_path_val = '/home/bdaniela/zero-shot-style/final_scatter_val.png'
    plot_graph_on_all_data(val_data, source_labels, tokenizer, device, fig_path_val, model, inner_batch_size, state='final',
                           batch_size=len(set(val_data['User'])))

    print('finished to plot.')


def evaluate(model,val_data, labels, tokenizer,batch_size,inner_batch_size):
    val = Dataset(val_data, labels, tokenizer,inner_batch_size)
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


def main():
    print('Start!')
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='description')
    parser.add_argument('--lr', type=float, default=1e-4, help='description')
    parser.add_argument('--margin', type=float, default=0.4, help='description')
    parser.add_argument('--batch_size', type=int, default=2, help='description')
    parser.add_argument('--inner_batch_size', type=int, default=50, help='description')
    parser.add_argument('--resume', type=str, default='allow', help='continue logging to run_id')
    parser.add_argument('--run_id', type=str, default=None, help='wandb run_id')
    parser.add_argument('--tags', type=str, nargs='+', default=None, help='wandb tags')
    parser.add_argument('--wandb_mode', type=str, default='online', help='disabled, offline, online')
    parser.add_argument('--data_file', type=str, default='', help='')
    args = parser.parse_args()
    config = vars(args)

    wandb.init(project='zero-shot-learning',
               config=config,
               resume=args.resume,
               id=args.run_id,
               mode=args.wandb_mode,
               tags=args.tags)
    EPOCHS = config['epochs']
    LR = config['lr']
    batch_size = config['batch_size']
    inner_batch_size = config['inner_batch_size']
    margin = config['margin']
    data_file = config['data_file']
    base_path = '~/zero-shot-style/'
    #datapath = 'preprocessed_data.csv'
    datapath = 'preprocessed_data.csv'
    df = pd.read_csv(base_path + datapath)
    df.head()

    df.groupby(['User']).size().plot.bar()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    labels = {}
    for i, label in enumerate(set(df.iloc[:, 0])):
        labels[label] = i

    np.random.seed(112)
    print('Splitting DB to train, val and test data frames.')
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    #print(len(df_train), len(df_val), len(df_test))


    model = BertClassifier()
    #train model
    train(model, df_train,df_val, LR, EPOCHS, labels, tokenizer,batch_size,margin,inner_batch_size)



    evaluate(model, df_test, labels, tokenizer,batch_size)
    print('  finish!')


if __name__=='__main__':
    main()