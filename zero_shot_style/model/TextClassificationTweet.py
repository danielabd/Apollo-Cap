import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import SGD
from tqdm import tqdm
import operator
import zero_shot_style.model.Mining as Mining
from zero_shot_style.model.Mining import *
from argparse import ArgumentParser
import wandb

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labels_set_dict, inner_batch_size):
        self.labels = [labels_set_dict[label] for label in df['label']] #create list of idxs for labels
        self.labels_set = list(set(self.labels))
        self.texts = df['text']#df['Tweet'] #[text for text in df['Tweet']]
        self.batch_size_per_label = inner_batch_size
        pass

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels_set)

    def __getitem__(self, item):
        label = self.labels_set[item]
        list_idxs_for_label = [np.array(self.labels) == label]
        full_tweets_list = list(operator.itemgetter(list_idxs_for_label)(np.array(self.texts)))
        batch_tweets = random.sample(full_tweets_list,min(len(full_tweets_list),self.batch_size_per_label))
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
    texts_list = []
    labels_list = []
    for list_for_label in data:
        for text in list_for_label[0]:
            texts_list.append(text)
            labels_list.append(list_for_label[1])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts_list = tokenizer(texts_list, padding='max_length', max_length=40, truncation=True,
                                     return_tensors="pt")
    return tokenized_texts_list, labels_list, texts_list

def plot_graph_on_all_data(df_data, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, batch_size, title):
    data_set = Dataset(df_data, labels_set_dict, inner_batch_size)
    data_dataloader = torch.utils.data.DataLoader(data_set, collate_fn=collate_fn, batch_size = batch_size, shuffle=True,
                                                   num_workers=0)
    x_train, y_train, text_list = next(iter(data_dataloader))
    x_train = x_train.to(device)
    model.to(device)
    y_train = torch.from_numpy(np.asarray(y_train)).to(device)
    outputs = model(x_train['input_ids'], x_train['attention_mask'])
    outputs = outputs.detach().cpu().numpy()
    labeldf = pd.DataFrame({'Label': [labels_idx_to_str[user_idx] for user_idx in y_train.cpu().numpy()]})
    embdf = pd.DataFrame(outputs, columns=[f'emb{i}' for i in range(outputs.shape[1])])
    textdf = pd.DataFrame({'text': text_list})
    all_data = pd.concat([labeldf, embdf, textdf], axis=1, ignore_index=True)
    all_data.columns = ['Label'] + [f'emb{i}' for i in range(outputs.shape[1])] + ['text']
    wandb.log({title: all_data})


def train(model, optimizer, df_train, df_val, labels_set_dict, labels_idx_to_str, epochs, batch_size, margin, inner_batch_size, path_for_saving_model):
    print('Starting to train...')
    val_batch_size_for_plot = len(set(df_val['label'])) #min(batch_size,len(set(df_val['label'])))# suppose that the first column is for label
    train_batch_size_for_plot = len(set(df_train['label'])) #min(batch_size,len(set(df_train['label'])))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #plot initial graphs
    plot_graph_on_all_data(df_train, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, train_batch_size_for_plot, "initial_train_text")
    plot_graph_on_all_data(df_val, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, val_batch_size_for_plot, "initial_val_text")

    train_data_set = Dataset(df_train,labels_set_dict, inner_batch_size)
    train_dataloader = torch.utils.data.DataLoader(train_data_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)

    if use_cuda:
        model = model.to(device)

    ####!todo: continue from here
    model.train()
    for epoch in range(epochs):
        running_loss = []
        for step, (tokenized_texts_list, labels, texts_list) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            labels = torch.from_numpy(np.asarray(labels)).to(device)
            masks = tokenized_texts_list['attention_mask'].to(device)
            input_ids = tokenized_texts_list['input_ids'].squeeze(1).to(device)
            outputs = model(input_ids, masks)

            #using tripletloss
            loss, num_positive_triplets, num_valid_triplets, all_triplet_loss_avg = Mining.online_mine_all(labels, outputs, margin,device=device)
            fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
            print("\nEpoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)),'\n')
            log_dict = {'train/epoch': epoch,
                        'train/train_loss': loss.cpu().detach().numpy(),
                        'train/fraction_positive_triplets': fraction_positive_triplets,
                        'train/num_positive_triplets': num_positive_triplets,
                        'train/all_triplet_loss_avg': all_triplet_loss_avg}
            wandb.log({"log_dict": log_dict})
            plot_graph_on_all_data(df_val, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, val_batch_size_for_plot,"val_text")
            if np.mod(epoch,10)==0:
                plot_graph_on_all_data(df_train, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size,
                                   train_batch_size_for_plot, "intermediate_train_text")

        # save model every epoch
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    'loss': loss
                    }, path_for_saving_model)




    print('Finished to train')
    #finally check on all data training
    # fig_path = '/home/bdaniela/zero-shot-style/final_scatter_train.png'
    plot_graph_on_all_data(df_train, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, train_batch_size_for_plot,"final_train_text")
    print('finished to train.')


def evaluate(model,df_test, labels_set_dict, labels_idx_to_str, batch_size,inner_batch_size):
    #evaluation on test set
    test = Dataset(df_test, labels_set_dict, inner_batch_size)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_results = []
    labels_results = []
    model.eval()
    with torch.no_grad():
        for tokenized_tweets_list, labels, text_tweet_list in tqdm(test_dataloader, desc="Testing", leave=False):
            labels = torch.from_numpy(np.asarray(labels)).to(device)
            masks = tokenized_tweets_list['attention_mask'].to(device)
            input_ids = tokenized_tweets_list['input_ids'].squeeze(1).to(device)
            outputs = model(input_ids, masks)
            test_results.append(outputs.cpu().numpy())
            # train_results.append(model(text.to(device)).cpu().numpy())
            labels_results.append(labels)

    test_results = np.concatenate(test_results)
    labels_results = np.concatenate(labels_results)

    test_batch_size_for_plot = len(set(df_test['User']))
    plot_graph_on_all_data(df_test, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, test_batch_size_for_plot,"test_text")
    print('finished to train.')

    # plt.figure(figsize=(15, 10), facecolor="azure")
    # for label in np.unique(labels):
    #     tmp = train_results[labels == label]
    #     plt.scatter(tmp[:, 0], tmp[:, 1], label=label)
    #
    # plt.legend()
    # plt.show()

# def create_correct_df(df,num_of_labels):
#     # labels_set_dict = {dmiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral}
#     labels_set = df.columns[-num_of_labels:]
#     #create new df
#     fixed_df = {'text':df['text']}
#     list_of_labels = []
#     fixed_list_of_texts = []
#     for i in range(df.shape[0]):#go over all rows
#         if i==100: #todo: remove it
#             break
#         relevant_idxs_for_labels =  np.where(df.iloc[i, -num_of_labels:].values == 1)
#         labels = labels_set[relevant_idxs_for_labels[0]]
#         for l in labels:
#             try:
#                 fixed_list_of_texts.append(df['text'][i])
#                 list_of_labels.append(l)
#             except:
#                 pass
#     fixed_df = {'label': list_of_labels, 'text': fixed_list_of_texts}
#     return fixed_df
def create_correct_df(df,num_of_labels,desired_labels):
    # labels_set_dict = {dmiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral}
    labels_set = df.columns[-num_of_labels:]
    #create new df
    fixed_df = {'text':df['text']}
    list_of_labels = []
    fixed_list_of_texts = []
    for i in range(df.shape[0]):#go over all rows
        if i==10000: #todo: remove it
            break
        relevant_idxs_for_labels =  np.where(df.iloc[i, -num_of_labels:].values == 1)
        if len(relevant_idxs_for_labels)>1:
            continue
        labels = labels_set[relevant_idxs_for_labels[0]]
        for l in labels:
            if l not in desired_labels:
                continue
            try:
                fixed_list_of_texts.append(df['text'][i])
                list_of_labels.append(l)
            except:
                pass
    fixed_df = pd.DataFrame({'label': list_of_labels, 'text': fixed_list_of_texts})
    return fixed_df


def main():
    print('Start!')
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='description')
    parser.add_argument('--lr', type=float, default=1e-4, help='description')
    parser.add_argument('--margin', type=float, default=0.4, help='description')
    parser.add_argument('--batch_size', type=int, default=4, help='description')
    parser.add_argument('--inner_batch_size', type=int, default=30, help='description')
    parser.add_argument('--resume', type=str, default='allow', help='continue logging to run_id')
    parser.add_argument('--run_id', type=str, default=None, help='wandb run_id')
    parser.add_argument('--tags', type=str, nargs='+', default=None, help='wandb tags')
    parser.add_argument('--wandb_mode', type=str, default='online', help='disabled, offline, online')
    parser.add_argument('--data_file', type=str, default='preprocessed_data.csv', help='')
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

    data_name = 'go_emotions'  # 'Twitter'
    if data_name=='go_emotions': #https://github.com/google-research/google-research/tree/master/goemotions
        data_file = ['goemotions_1.csv','goemotions_2.csv','goemotions_3.csv']


    base_path = '/home/bdaniela/zero-shot-style/zero_shot_style/model/data'

    if type(data_file)!=list:
        datapath = os.path.join(base_path, data_file)
        s_df = pd.read_csv(datapath)
    else:
        s_df = pd.read_csv(os.path.join(base_path, data_file[0]))
        for f in data_file[1:]:
            datapath = os.path.join(base_path, f)
            cur_df = pd.read_csv(datapath)
            s_df = pd.concat([s_df, cur_df], axis=0, ignore_index=True)

    load_model = True

    path_for_saving_model = os.path.join(base_path,"trained_model_emotions.pth")
    # datapath = os.path.join(base_path,data_file)
    # df = pd.read_csv(datapath)
    s_df.head()
    #df.groupby(['User']).size().plot.bar()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    num_of_labels = 28
    desired_labels = ['anger','caring','optimism','love']
    df = create_correct_df(s_df, num_of_labels ,desired_labels)

    np.random.seed(112)
    # print('Splitting DB to train, val and test data frames.')
    # s_df_train, s_df_val, s_df_test = np.split(df.sample(frac=1, random_state=42),
    #                                      [int(.8 * len(df)), int(.9 * len(df))])
    # print(len(s_df_train), len(s_df_val), len(s_df_test))

    print('Splitting DB to train, val and test data frames.')
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])
    print(len(df_train), len(df_val), len(df_test))


    # if data_name == 'go_emotions':
    #     df_train = create_correct_df(s_df_train,num_of_labels)
    #     df_val = create_correct_df(s_df_val,num_of_labels)
    #     df_test = create_correct_df(s_df_test,num_of_labels)

           #list_of_labels.append(df[])
    #
    #
    #     labels_set_dict = {}
    #     labels_idx_to_str = {}
    #     for i, label in enumerate(df.columns()[-28:]):
    #         labels_set_dict[label] = i
    #         labels_idx_to_str[i] = label
    #
    # elif data_name == 'Twitter':
    #     # users in Twitter
    #     labels_set_dict = {}
    #     labels_idx_to_str = {}
    #     for i, label in enumerate(set(df_train.iloc[:, 0])):
    #         labels_set_dict[label] = i
    #         labels_idx_to_str[i] = label


    if load_model:
        model = BertClassifier()
        optimizer = SGD(model.parameters(), lr=LR)
        checkpoint = torch.load(path_for_saving_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        last_loss = checkpoint['loss']
        model.train()

    else: #train from scratch
        model = BertClassifier()
        optimizer = SGD(model.parameters(), lr=LR)


    labels_set_dict = {}
    labels_idx_to_str = {}
    # for i, label in enumerate(set(df['label'])):
    for i, label in enumerate(s_df.columns[-num_of_labels:]):
        labels_set_dict[label] = i
        labels_idx_to_str[i] = label

    #train model
    train(model, optimizer, df_train, df_val, labels_set_dict, labels_idx_to_str, EPOCHS, batch_size,margin,inner_batch_size, path_for_saving_model)

    evaluate(model, df_test, labels_set_dict, labels_idx_to_str, batch_size, inner_batch_size)
    print('  finish!')


if __name__=='__main__':
    main()