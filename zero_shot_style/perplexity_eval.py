#https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk
import os.path
import pickle
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from zero_shot_style.datasets.senticap_reader import *
import numpy as np


def get_data(dataset_name,type_set):
    '''
    :param type_set: strind of train/val/test
    :param dataset_name: string with the name of the dataset. 'senticap'/'flickrstyle10k'
    :return: train_sentences: list of all sentences in train set
    '''
    data_dir = os.path.join(os.path.expanduser('~'),'data')
    train_data_path = os.path.join(data_dir, dataset_name, 'annotations', type_set+'.pkl')
    sentences = []
    with open(train_data_path, 'rb') as r:
        data = pickle.load(r)
    for k in data:
        sentences.extend(data[k]['factual'])
        sentences.extend(data[k]['humor'])
        sentences.extend(data[k]['romantic'])
    return sentences


def train_perplexity_model(train_sentences,n):
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent)))
                      for sent in train_sentences]
    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
    model = MLE(n)
    model.fit(train_data, padded_vocab)
    return model



def calc_perplexity_per_setence(sentence,n, pp_model):
    tokenized_text = list(map(str.lower, nltk.tokenize.word_tokenize(sentence)))
    # test_data, _ = padded_everygram_pipeline(n, tokenized_text)
    # for test in test_data:
    #    print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])
    test_data, _ = padded_everygram_pipeline(n, tokenized_text)
    pp_scores = []
    for i, test in enumerate(test_data):
        pp_scores.append(pp_model.perplexity(test))
    if np.inf in pp_scores:
        pp_scores_wo_inf = [e for e in pp_scores if e != np.inf]
    else:
        pp_scores_wo_inf = pp_scores
    valid_sentences_percent = len(pp_scores_wo_inf)/pp_scores # pp!=inf
    avg_pp_score = np.mean(pp_scores_wo_inf)
    print(f'Average perplexity score for test set is: {avg_pp_score}. There are {valid_sentences_percent}% valid sentences (pp!=inf)')
    return avg_pp_score, valid_sentences_percent

def calc_perplexity(test_sentences,n, pp_model):
    #test_sentences = ['an apple', 'an ant']
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent)))
                      for sent in test_sentences]
    # test_data, _ = padded_everygram_pipeline(n, tokenized_text)
    # for test in test_data:
    #    print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])
    test_data, _ = padded_everygram_pipeline(n, tokenized_text)
    pp_scores = []
    for i, test in enumerate(test_data):
        pp_scores.append(pp_model.perplexity(test))
    if np.inf in pp_scores:
        pp_scores_wo_inf = [e for e in pp_scores if e != np.inf]
    else:
        pp_scores_wo_inf = pp_scores
    valid_sentences_percent = len(pp_scores_wo_inf)/pp_scores # pp!=inf
    avg_pp_score = np.mean(pp_scores_wo_inf)
    print(f'Average perplexity score for test set is: {avg_pp_score}. There are {valid_sentences_percent}% valid sentences (pp!=inf)')
    return avg_pp_score, valid_sentences_percent


def main():
    dataset_name = 'flickrstyle10k' #'flickrstyle10k'/'senticap'
    n=3 # MSCap used n=3
    train_sentences = get_data(dataset_name,'train')
    train_sentences.extend(get_data(dataset_name,'val'))
    test_sentences = get_data(dataset_name,'test')
    pp_model = train_perplexity_model(train_sentences, n)
    avg_pp_score, valid_sentences_percent = avg_pp_score = calc_perplexity(test_sentences, n, pp_model)
    print('finish')

if __name__=='__main__':
    main()