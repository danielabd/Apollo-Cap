import pandas as pd
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import SGD
from tqdm import tqdm
import operator
from zero_shot_style.model.Mining import *
from argparse import ArgumentParser
import wandb

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labels_set_dict, inner_batch_size):
        self.labels = [labels_set_dict[label] for label in df['User']]
        self.labels_set = list(set(self.labels))
        self.texts = df['Tweet'] #[text for text in df['Tweet']]
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
    tweets_list = []
    labels_list = []
    for list_for_label in data:
        for tweet in list_for_label[0]:
            tweets_list.append(tweet)
            labels_list.append(list_for_label[1])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_tweets_list = tokenizer(tweets_list, padding='max_length', max_length=512, truncation=True,
                                     return_tensors="pt")
    return tokenized_tweets_list, labels_list, tweets_list

def plot_graph_on_all_data(df_data, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, batch_size, title):
    data_set = Dataset(df_data, labels_set_dict, inner_batch_size)
    data_dataloader = torch.utils.data.DataLoader(data_set, collate_fn=collate_fn, batch_size = batch_size, shuffle=True,
                                                   num_workers=0)
    x_train, y_train, text_tweet_list  = next(iter(data_dataloader))
    x_train = x_train.to(device)
    model.to(device)
    y_train = torch.from_numpy(np.asarray(y_train)).to(device)
    outputs = model(x_train['input_ids'], x_train['attention_mask'])
    outputs = outputs.detach().cpu().numpy()
    userdf = pd.DataFrame({'User': [labels_idx_to_str[user_idx] for user_idx in y_train.cpu().numpy()]})
    embdf = pd.DataFrame(outputs, columns=[f'emb{i}' for i in range(outputs.shape[1])])
    textdf = pd.DataFrame({'text': text_tweet_list})
    all_data = pd.concat([userdf, embdf, textdf], axis=1, ignore_index=True)
    all_data.columns = ['User'] + [f'emb{i}' for i in range(outputs.shape[1])] + ['text']
    wandb.log({title: all_data})


def train(model, optimizer, df_train, df_val, labels_set_dict, labels_idx_to_str, epochs, batch_size, margin, inner_batch_size, path_for_saving_model):
    print('Starting to train...')
    val_batch_size_for_plot = len(set(df_val['User']))
    train_batch_size_for_plot = len(set(df_train['User']))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #plot initial graphs
    plot_graph_on_all_data(df_train, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, train_batch_size_for_plot,"initial_train_text")
    plot_graph_on_all_data(df_val, labels_set_dict, labels_idx_to_str, device, model, inner_batch_size, val_batch_size_for_plot, "initial_val_text")

    train_data_set = Dataset(df_train,labels_set_dict, inner_batch_size)
    train_dataloader = torch.utils.data.DataLoader(train_data_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)

    if use_cuda:
        model = model.to(device)

    model.train()
    for epoch in range(epochs):
        running_loss = []
        for step, (tokenized_tweets_list, labels, text_tweet_list) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            labels = torch.from_numpy(np.asarray(labels)).to(device)
            masks = tokenized_tweets_list['attention_mask'].to(device)
            input_ids = tokenized_tweets_list['input_ids'].squeeze(1).to(device)
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


def main():
    print('Start!')
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='description')
    parser.add_argument('--lr', type=float, default=1e-4, help='description')
    parser.add_argument('--margin', type=float, default=0.4, help='description')
    parser.add_argument('--batch_size', type=int, default=6, help='description')
    parser.add_argument('--inner_batch_size', type=int, default=5, help='description')
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

    load_model = True
    base_path = '/home/bdaniela/zero-shot-style/data'
    path_for_saving_model = os.path.join(base_path,"trained_model.pth")
    datapath = os.path.join(base_path,data_file)
    df = pd.read_csv(datapath)
    df.head()


    df.groupby(['User']).size().plot.bar()

    np.random.seed(112)
    print('Splitting DB to train, val and test data frames.')
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    print(len(df_train), len(df_val), len(df_test))

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
    for i, label in enumerate(set(df_train.iloc[:, 0])):
        labels_set_dict[label] = i
        labels_idx_to_str[i] = label

    #train model
    train(model, optimizer, df_train, df_val, labels_set_dict, labels_idx_to_str, EPOCHS, batch_size,margin,inner_batch_size, path_for_saving_model)

    evaluate(model, df_test, labels_set_dict, labels_idx_to_str, batch_size, inner_batch_size)
    print('  finish!')



    # if __name__=='__main__':
    # main()