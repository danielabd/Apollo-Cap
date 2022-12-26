import nltk
import torch
import os
import pickle
import csv
from datetime import datetime
import pandas as pd

# Now we'll define our n-gram model as a PyTorch module
class NGramModel(torch.nn.Module):
    def __init__(self, vocab_size, n):
        super().__init__()
        self.probs = torch.nn.Parameter(torch.ones((vocab_size,) * n) / vocab_size ** (n - 1))
        self.n = n

    def forward(self, x):
        # The input x is a tensor of token indices
        # Shift the input tensor to the left to get the previous tokens
        prev_tokens = [torch.cat((x[:1], x[:-1]))]
        for i in range(1, self.n - 1):
            prev_tokens.append(torch.cat((prev_tokens[-1][:1], prev_tokens[-1][:-1])))

        # Create a tuple of the previous tokens and the current token
        ngram_tokens = tuple(prev_tokens + [x])

        # Use the tuple of tokens as indices to look up the probability in the probs tensor
        return self.probs[ngram_tokens]


# Once the model is trained, you can use it to evaluate the perplexity of new sentences
def evaluate_perplexity(model, sentence):
    # Convert the sentence into a tensor of token indices
    input



def get_gts_data(test_set_path):
    gts_per_data_set = {}
    for dataset_name in test_set_path:
        gts = {}
        with open(test_set_path[dataset_name], 'rb') as r:
            data = pickle.load(r)
        for k in data:
            gts[k] = {}
            gts[k]['factual'] = data[k]['factual']  #todo: check if there is need to concatenate factual from senticap and flickrstyle10k
            gts[k]['img_path'] = data[k]['image_path']
            if dataset_name == 'flickrstyle10k':
                gts[k]['humor'] = data[k]['humor']
                gts[k]['romantic'] = data[k]['romantic']
            elif dataset_name == 'senticap':
                gts[k]['positive'] = data[k]['positive']
                gts[k]['negative'] = data[k]['negative']
        gts_per_data_set[dataset_name] = gts
    return gts_per_data_set


def get_res_data(res_paths):
    '''

    :param res_paths: dict. keys:  'prompt_manipulation', 'image_manipulation'. values: path to res
    :return: res_data_per_test: dict. keys:  'prompt_manipulation', 'image_manipulation'. values: dict to res per image name and style
    '''
    res_data_per_test = {}
    for test_type in res_paths:
        res_data = {}
        with open(res_paths[test_type], 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            title = True
            for row in spamreader:
                if '.jpg' in row[0]:
                    k = row[0].split('.jpg')[0]
                else:
                    k = row[0]
                if title:
                    title = False
                    continue
                else:
                    try:
                        res_data[k]={}
                        res_data[k]['factual'] = row[1]
                        res_data[k]['positive'] = row[2]
                        res_data[k]['negative'] = row[3]
                        res_data[k]['romantic'] = row[4]
                        res_data[k]['humor'] = row[5]
                    except:
                        pass
        res_data_per_test[test_type] = res_data
    return res_data_per_test

def main():
    cuda_idx = "1"
    styles = ['factual','positive', 'negative', 'humor', 'romantic']
    data_dir = os.path.join(os.path.expanduser('~'), 'data')
    gt_imgs_for_test = os.path.join(data_dir, 'gt_imgs_for_test')
    #path_test_prompt_manipulation = os.path.join(os.path.expanduser('~'),'results','04_15_54__14_12_2022','results_all_models_source_classes_04_15_54__14_12_2022.csv')
    #path_test_image_manipulation = os.path.join(os.path.expanduser('~'),'results','11_45_38__14_12_2022','results_all_models_source_classes_11_45_38__14_12_2022.csv')
    path_test_prompt_manipulation = os.path.join(os.path.expanduser('~'),'results','prompt_manipulation_01_31_38__19_12_2022','prompt_manipulation_01_31_38__19_12_2022.csv')
    path_test_image_manipulation = os.path.join(os.path.expanduser('~'),'results','image_manipulation_01_23_57__19_12_2022','image_manipulation_results_all_models_source_classes_01_23_57__19_12_2022.csv')
    txt_cls_model_path = os.path.join(os.path.expanduser('~'),'checkpoints','best_model','best_text_style_classification_model.pth')
    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    label = '25_12_2022_v1' # cur_time
    eval_results_path = os.path.join(data_dir,label+'_eval_results.csv')


    res_paths = {}
    res_paths['prompt_manipulation'] = path_test_prompt_manipulation
    res_paths['image_manipulation'] = path_test_image_manipulation

    dataset_names =['senticap', 'flickrstyle10k']
    #metrics = ['bleu','rouge', 'CLIPScoreRef','CLIPScore','style_classification', 'fluency']   # ['bleu','rouge','meteor', 'spice', 'CLIPScoreRef','CLIPScore','style_classification', 'fluency']
    metrics = ['fluency']   # ['bleu','rouge','meteor', 'spice', 'CLIPScoreRef','CLIPScore','style_classification', 'fluency']
    ngram_for_fluency = 3  # MSCap used n=3
    num_epochs =3
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_train =df_train[:10]
    source_sentences = list(df_train['text'])
    labels_dict_idxs = {}
    for i, label in enumerate(list(set(list(df_train['category'])))):
        labels_dict_idxs[label] = i

    test_set_path = {}
    for dataset_name in dataset_names:
        test_set_path[dataset_name] = os.path.join(data_dir, dataset_name, 'annotations', 'test.pkl')
    gts_per_data_set = get_gts_data(test_set_path)

    res_data_per_test = get_res_data(res_paths)

    # Assume that we have a dataset of sentences stored in a list called "sentences"

    # First, we'll build a vocabulary of all the unique tokens in the dataset


    sentences = [list(map(str.lower, nltk.tokenize.word_tokenize(sent)))
                      for sent in source_sentences]
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence)

    # Next, we'll convert the vocabulary into a list and create a mapping from tokens to indices
    vocab = list(vocab)
    token_to_index = {token: index for index, token in enumerate(vocab)}

    # Create an instance of the model and define the loss function and optimizer
    model = NGramModel(len(vocab), 3)  # This creates a 3-gram model
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train the model by iterating over the training set and updating the model's parameters
    for epoch in range(num_epochs):
        for sentence in sentences:
            # Convert the sentence into a tensor of token indices
            input_tensor = torch.tensor([token_to_index[token] for token in sentence])
            input_tensor = input_tensor.unsqueeze(1)

            # Calculate the log probabilities of each token
            log_probs = torch.log(model(input_tensor))
            #log_probs = log_probs.unsqueeze(1)
            # Calculate the loss
            loss = loss_fn(log_probs, input_tensor)

            # Zero the gradients and backpropagate the loss
            optimizer.zero_grad()
            loss.backward()

            # Update the model's parameters
            optimizer.step()

    #evaluate_perplexity(model, sentence)
    print('finish')

if __name__=='__main__':
    main()