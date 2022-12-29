#try to use it: for adapting to python3
#https: // github.com / sks3i / pycocoevalcap
#https://github.com/wangleihitcs/CaptionMetrics
#from nltk.translate.bleu_score import sentence_bleu
#from nltk.translate import meteor
#from nltk import word_tokenize
#import os
#import pickle
import shutil
import timeit
from datetime import datetime
from evaluate import load

import pandas as pd
import torch
import csv
import numpy as np
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import os
import pickle
import style_cls
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice import Spice
from zero_shot_style.model.ZeroCLIP import CLIPTextGenerator
from text_style_classification import evaluate as evaluate_text_style_classification
from text_style_classification import BertClassifier, tokenizer
MAX_PP_SCORE = 100
NORMALIZE_GRADE_SCALE = 100

class CLIPScoreRef:
    def __init__(self,text_generator):
        self.text_generator = text_generator

    def compute_score(self,gts, res):
        '''

        :param gts: list of str
        :param res: str
        :return:
        '''
        #print("calculate CLIPScoreRef...")
        scores_for_all = []
        for k in gts:
            for gt in gts[k]:
                text_features_gt = self.text_generator.get_txt_features(gt)
                text_features_ref = self.text_generator.get_txt_features(list(res.values())[0])
                with torch.no_grad():
                    clip_score_ref = (text_features_ref @ text_features_gt.T)
                    score = clip_score_ref.cpu().numpy()
                scores_for_all.append(score)
        avg_score = np.mean(scores_for_all)
        #print('CLIPScoreRef = %s' % avg_score)
        return avg_score, scores_for_all

class CLIPScore:
    def __init__(self,text_generator):
        self.text_generator = text_generator

    def compute_score(self,img_path, res):
        '''

        :param img_path: str full path to the image
        :param res: str
        :return:
        '''
        #print("calculate CLIPScore...")
        image_features = self.text_generator.get_img_feature([img_path], None)
        text_features = self.text_generator.get_txt_features(res)
        with torch.no_grad():
            clip_score = (image_features @ text_features.T)
        score = clip_score.cpu().numpy()
        #print('CLIPScore = %s' % score)
        return score[0][0], [score]

class STYLE_CLS:
    def __init__(self, txt_cls_model_paths, data_dir, desired_cuda_num, labels_dict_idxs):
        self.data_dir = data_dir
        self.desired_cuda_num = desired_cuda_num
        self.labels_dict_idxs = labels_dict_idxs
        self.df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        use_cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{desired_cuda_num}" if use_cuda else "cpu")  # todo: remove
        self.model = {}
        for dataset_name in txt_cls_model_paths:
            self.model[dataset_name] = self.load_model(txt_cls_model_paths[dataset_name])


    def load_model(self, txt_cls_model_path):
        model = BertClassifier()
        checkpoint = torch.load(txt_cls_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model


    def compute_score(self, res, gt_label, dataset_name):
        '''

        :param gts: list of text
        :param res: dict. key=str. value=list of single str
        :return:
        '''

        res_tokens = tokenizer(list(res.values())[0][0], padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt")
        gt_label_idx = torch.tensor(self.labels_dict_idxs[gt_label]).to(self.device)
        mask = res_tokens['attention_mask'].to(self.device)
        input_id = res_tokens['input_ids'].squeeze(1).to(self.device)
        output = self.model[dataset_name](input_id, mask)
        cls_score = output.argmax(dim=1) == gt_label_idx
        cls_score_np = int(cls_score.cpu().data.numpy())
        return cls_score_np, None

    def compute_score_for_total_data(self, gts, res, dataset_name):
        total_acc_test_for_all_data = evaluate_text_style_classification(self.model[dataset_name], self.df_test, self.labels_dict_idxs, self.desired_cuda_num)
        return total_acc_test_for_all_data, None

class Fluency:
    def __init__(self):
        self.model_id = 'gpt2'
        self.perplexity = load("perplexity", module_type="measurement")
        self.tests = []
        self.metric = []
        self.k = []
        self.style = []

    def reset_keys(self):
        self.tests = []
        self.metric = []
        self.k = []
        self.style = []

    def add_test(self, res, metric, k, style):
        self.tests.append(list(res.values())[0][0])
        self.metric.append(metric)
        self.k.append(k)
        self.style.append(style)

    def compute_score(self, score_dict_per_metric, score_per_metric_and_style, all_scores, test_type):
        '''
        sentence = list(res.values())[0]
        results = self.perplexity.compute(data=sentence, model_id=self.model_id, add_start_token=False)
        return results['mean_perplexity'], results['perplexities']
        '''
        results = self.perplexity.compute(data=self.tests, model_id=self.model_id, add_start_token=False)
        for i,pp in enumerate(results['perplexities']):
            k = self.k[i]
            style = self.style[i]
            metric = self.metric[i]
            score_dict_per_metric[metric][k][style] = pp
            score_per_metric_and_style[metric][style].append(score_dict_per_metric[metric][k][style])
            all_scores = save_all_data_k(all_scores, k, test_type, style, metric, score_dict_per_metric, res=self.tests[i])# save all data per key frames
        return score_dict_per_metric,score_per_metric_and_style, all_scores

'''
class Fluency:
    def __init__(self,n=3):
        self.n = n
        self.pp_model = None

    def train_pp_model(self,train_sentences):
        tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent)))
                          for sent in train_sentences]
        train_data, padded_vocab = padded_everygram_pipeline(self.n, tokenized_text)
        self.pp_model = MLE(self.n)
        self.pp_model.fit(train_data, padded_vocab)

    def compute_score(self, gts, sentence_dict):
        sentence = list(sentence_dict.values())[0][0]
        tokenized_text = list(map(str.lower, nltk.tokenize.word_tokenize(sentence)))

        #test_sentences = ['A skier is working his way down the rough hill.','An old abandoned building with a clock has been eclipsed by the stained glass splendor of an evil corporations skyscraper.']
        #tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent)))
        #                  for sent in test_sentences]

        # test_data, _ = padded_everygram_pipeline(n, tokenized_text)
        # for test in test_data:
        #    print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])


        test_data, _ = padded_everygram_pipeline(self.n, tokenized_text)
        pp_scores = []
        for i, test in enumerate(test_data):
            pp_scores.append(self.pp_model.perplexity(test))
        if np.inf in pp_scores:
            pp_scores_wo_inf = [e for e in pp_scores if e != np.inf]
        else:
            pp_scores_wo_inf = pp_scores
        valid_sentences_percent = len(pp_scores_wo_inf) / len(pp_scores)*100  # pp!=inf
        if valid_sentences_percent>0:
            avg_pp_score = np.mean(pp_scores_wo_inf)
        else:
            avg_pp_score = MAX_PP_SCORE
        print(
            f'Average perplexity score for test set is: {avg_pp_score}. There are {valid_sentences_percent}% valid sentences (pp!=inf)')
        return avg_pp_score, valid_sentences_percent
'''

def save_all_data_k(all_scores, k, test_type, style, metric, score_dict_per_metric, res = None, gts = None, img_path = None):
    # save all data per key frames
    if k not in all_scores:
        all_scores[k] = {}
    if test_type not in all_scores[k]:
        all_scores[k][test_type] = {}
    if style not in all_scores[k][test_type]:
        all_scores[k][test_type][style] = {}

    all_scores[k][test_type][style][metric] = score_dict_per_metric[metric][k][style]

    if res and ('res' not in all_scores[k][test_type][style]):
        all_scores[k][test_type][style]['res'] = res
    if gts and ('gts' not in all_scores[k][test_type][style]):
        all_scores[k][test_type][style]['gts'] = gts
    if img_path and ('img_path' not in all_scores[k][test_type][style]):
        all_scores[k][test_type][style]['img_path'] = img_path
    return all_scores



def calc_score(gts_per_data_set, res, styles, metrics, cuda_idx, data_dir, txt_cls_model_paths_to_load, labels_dict_idxs, gt_imgs_for_test, styles_per_dataset):
    print("Calculate scores...")

    mean_score = {}
    if ('CLIPScoreRef' in metrics) or ('CLIPScore'in metrics):
        text_generator = CLIPTextGenerator(cuda_idx=cuda_idx)
    if 'style_classification' in metrics:
        style_cls_obj = STYLE_CLS(txt_cls_model_paths_to_load, data_dir, cuda_idx, labels_dict_idxs)
    if 'fluency' in metrics:
        fluency_obj = Fluency()
    all_scores = {}
    for test_type in res:
        print(f"Calc scores for experiment: **** {test_type} *****")
        mean_score_per_dataset = {}
        for dataset_name in gts_per_data_set:
            print(f"Calc scores for dataset: **** {dataset_name} *****")
            score_per_metric_and_style = {}
            score_dict_per_metric = {}
            scores_dict_per_metric = {}
            mean_score_per_metric_and_style = {}
            for metric in metrics:
                print(f"    Calc scores for metric: ***{metric}***")

                if metric == 'bleu':
                    scorer = Bleu(n=4)
                elif metric == 'cider':
                    scorer = Cider()
                elif metric == 'meteor':
                    scorer = Meteor()
                elif metric == 'rouge':
                    scorer = Rouge()
                elif metric == 'spice':
                    scorer = Spice()
                elif metric == 'CLIPScoreRef':
                    scorer = CLIPScoreRef(text_generator)
                elif metric == 'CLIPScore':
                    scorer = CLIPScore(text_generator)
                elif metric == 'fluency':
                    scorer = fluency_obj
                    scorer.reset_keys()
                elif metric == 'style_classification':
                    scorer = style_cls_obj

                score_per_metric_and_style[metric] = {}
                for style in styles:
                    score_per_metric_and_style[metric][style] = []
                score_dict_per_metric[metric] = {}
                scores_dict_per_metric[metric] = {}
                mean_score_per_metric_and_style[metric] = {}
                for i1,k in enumerate(res[test_type]):
                    if k in gts_per_data_set[dataset_name]:
                        score_dict_per_metric[metric][k] = {}
                        scores_dict_per_metric[metric][k] = {}
                        for i2,style in enumerate(styles):
                            if style not in styles_per_dataset[dataset_name]:
                                continue
                            if style == 'factual' and metric == 'style_classification':
                                continue
                            if style in gts_per_data_set[dataset_name][k] and style in res[test_type][k]:
                                if not gts_per_data_set[dataset_name][k][style]:
                                    continue
                                tmp_res = {k: [res[test_type][k][style]]}
                                if metric == 'CLIPScore':
                                    gts_per_data_set[dataset_name][k]['img_path'] = os.path.join(gt_imgs_for_test,gts_per_data_set[dataset_name][k]['img_path'].split('/')[-1])
                                    score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][style] = scorer.compute_score(gts_per_data_set[dataset_name][k]['img_path'], tmp_res)
                                    score_per_metric_and_style[metric][style].append(
                                        score_dict_per_metric[metric][k][style])
                                    all_scores = save_all_data_k(all_scores, k, test_type, style, metric, score_dict_per_metric, res=tmp_res[k][0], img_path = gts_per_data_set[dataset_name][k]['img_path'])
                                elif metric == 'style_classification':
                                    score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][style] = scorer.compute_score(tmp_res,style,dataset_name)
                                    score_per_metric_and_style[metric][style].append(
                                        score_dict_per_metric[metric][k][style])
                                    all_scores = save_all_data_k(all_scores, k, test_type, style, metric, score_dict_per_metric, res=tmp_res[k][0])
                                elif metric == 'fluency':
                                    scorer.add_test(tmp_res, metric,k,style)
                                else:
                                    tmp_gts = {k: gts_per_data_set[dataset_name][k][style]}
                                    score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][
                                        style] = scorer.compute_score(tmp_gts, tmp_res)
                                    score_dict_per_metric[metric][k][style] = np.mean(score_dict_per_metric[metric][k][style])#todo: check if need it
                                    score_per_metric_and_style[metric][style].append(
                                        score_dict_per_metric[metric][k][style])
                                    all_scores = save_all_data_k(all_scores, k,test_type,style,metric, score_dict_per_metric, res=tmp_res[k][0], gts=tmp_gts[k])
                if metric == 'fluency':
                    score_dict_per_metric,score_per_metric_and_style, all_scores = fluency_obj.compute_score(score_dict_per_metric,score_per_metric_and_style, all_scores, test_type)

                for style in styles:
                    if mean_score_per_dataset == 'fluency':
                        mean_score_per_metric_and_style[metric][style] = np.mean(score_per_metric_and_style[metric][style])
                    else:
                        mean_score_per_metric_and_style[metric][style] = np.mean(score_per_metric_and_style[metric][style])*NORMALIZE_GRADE_SCALE
            mean_score_per_dataset[dataset_name] = mean_score_per_metric_and_style
        mean_score[test_type] = mean_score_per_dataset
    return mean_score, all_scores


def copy_imgs_to_test_dir(gts_per_data_set, res, styles, metrics, gt_imgs_for_test):
    print("Calculate scores...")
    imgs2cpy = []
    for test_type in res:
        for dataset_name in gts_per_data_set:
            for metric in metrics:
                for i1, k in enumerate(res[test_type]):
                    if k in gts_per_data_set[dataset_name]:
                        for i2, style in enumerate(styles):
                            if style in gts_per_data_set[dataset_name][k] and style in res[test_type][k]:
                                if not gts_per_data_set[dataset_name][k][style]:
                                    continue
                                if metric == 'CLIPScore':
                                    imgs2cpy.append(gts_per_data_set[dataset_name][k]['img_path'])

    for i in imgs2cpy:
        shutil.copyfile(i,os.path.join(gt_imgs_for_test,i.split('/')[-1]))
    return 0


'''
def bleu(gts, res,styles):
    print("Calculate bleu score...")
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score_per_style = {}
    for style in styles:
        score_per_style[style] = []
    score_dict = {}
    scores_dict = {}
    for k in res:
        score_dict[k] = {}
        scores_dict[k] = {}
        for style in styles:
            score_dict[k][style], scores_dict[k][style] = scorer.compute_score(gts[k][style], res[k][style])
            score_per_style[style].append(score_dict[k][style])

    #score, scores = scorer.compute_score(gts, res)
    mean_score_per_style = {}
    for style in styles:
        mean_score_per_style[style] = np.mean(score_per_style[style])
        #print(f'belu for {style} = {mean_score_per_style[style]}')
    return mean_score_per_style

def cider(gts, res):
    print("Calculate cider score...")
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('cider = %s' % score)
    return score


def meteor(gts, res):
    print("Calculate meteor score...")
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('meter = %s' % score)
    return score

def rouge(gts, res):
    print("Calculate rouge score...")
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)
    return score

def spice(gts, res):
    print("Calculate spice score...")
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('spice = %s' % score)
    return score

def CLIPScoreRef(res,gts,text_generator):
    print("calculate CLIPScoreRef...")
    scores_for_all = []
    scores_k = []
    for k in res.keys():
        for candidate_txt in gts[k]:
            text_features_gts = text_generator.get_txt_features(candidate_txt)
            text_features_ref = text_generator.get_txt_features(res[k])
            with torch.no_grad():
                clip_score_ref = (text_features_ref @ text_features_gts.T)
                scores_k.append(clip_score_ref.cpu().numpy())
        scores_for_all.append(np.mean(scores_k))
    avg_score = np.mean(scores_for_all)
    print('CLIPScoreRef = %s' % avg_score)
    return avg_score


def CLIPScore(text_generator, img_path, res):
    print("calculate CLIPScoreRef...")
    scores = []
    for k in res.keys():
        image_features = text_generator.get_img_feature([img_path], None)
        text_features = text_generator.get_txt_features(res[k])
        with torch.no_grad():
            clip_score = (image_features @ text_features.T)
        scores.append(clip_score.cpu().numpy())
    avg_score = np.mean(scores)
    print('CLIPScore = %s' % avg_score)
    return avg_score

def get_fluency_obj(data_dir,dataset_names, ngram_for_fluency):
    print("Train fluency model...")
    train_sentences = []
    for dataset_name in dataset_names:
        train_sentences.extend(get_all_sentences(data_dir, dataset_name,'train'))
        train_sentences.extend(get_all_sentences(data_dir, dataset_name,'val'))
    fluency_obj = Fluency(ngram_for_fluency)
    fluency_obj.train_pp_model(train_sentences)
    print("Finished to train fluency model.")
    return fluency_obj
'''

def style_accuracy():
    pass


def diversitiy(res,gts):
    print("Calculate vocabulary size...")
    vocab_size = {}
    for test_type in res:
        vocab_list = []
        for k in res[test_type]:
            for style in res[test_type][k]:
                tokenized_text = list(map(str.lower, nltk.tokenize.word_tokenize(res[test_type][k][style])))
                vocab_list.extend(tokenized_text)
                vocab_list = list(set(vocab_list))
        vocab_size[test_type] = len(vocab_list)
    #print(f'Vocabulary size is: {vocab_size}')
    return vocab_size



def get_all_sentences(data_dir, dataset_name,type_set):
    '''
    :param type_set: strind of train/val/test
    :param dataset_name: string with the name of the dataset. 'senticap'/'flickrstyle10k'
    :return: train_sentences: list of all sentences in train set
    '''
    data_path = os.path.join(data_dir, dataset_name, 'annotations', type_set+'.pkl')
    sentences = []
    with open(data_path, 'rb') as r:
        data = pickle.load(r)
    for k in data:
        sentences.extend(data[k]['factual'])
        if dataset_name == 'senticap':
            sentences.extend(data[k]['positive'])
            sentences.extend(data[k]['negative'])
        if dataset_name == 'flickrstyle10k':
            sentences.extend(data[k]['humor'])
            sentences.extend(data[k]['romantic'])
    return sentences


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


def write_results_for_all_frames(all_scores, tgt_eval_results_path_for_all_frames, metrics):
    print(f"Write results to {tgt_eval_results_path_for_all_frames}...")
    title = ['k','test_type','style']
    title.extend(metrics)
    title.extend(['res','gts','img_path'])
    with open(tgt_eval_results_path_for_all_frames, 'w') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(title)
        for k in all_scores:
            for test_type in all_scores[k]:
                for style in all_scores[k][test_type]:
                    row = [k, test_type, style]
                    for metric in metrics:
                        if metric in all_scores[k][test_type][style]:
                            row.append(all_scores[k][test_type][style][metric])
                        else:
                            row.append('')
                    if 'res' in all_scores[k][test_type][style]:
                        row.append(all_scores[k][test_type][style]['res'])
                    else:
                        row.append('')
                    if 'gts' in all_scores[k][test_type][style]:
                        row.append(all_scores[k][test_type][style]['gts'])
                    else:
                        row.append('')
                    if 'img_path' in all_scores[k][test_type][style]:
                        row.append(all_scores[k][test_type][style]['img_path'])
                    else:
                        row.append('')
                    writer.writerow(row)
    print(f"finished to write results for all frames in {tgt_eval_results_path_for_all_frames}")


def write_results(mean_score, tgt_eval_results_path,dataset_names, metrics, styles, vocab_size):
    print(f"Write results to {tgt_eval_results_path}...")
    with open(tgt_eval_results_path, 'w') as results_file:
        writer = csv.writer(results_file)
        dataset_row = ['Dataset']
        style_row = ['Style']
        metrics_row = ['Model\Metric']
        '''
        for dataset_name in dataset_names:
            for style in styles:
                for metric in metrics:
                    dataset_row.append(dataset_name)
                    style_row.append(style)
                    metrics_row.append(metric)
        '''
        total_rows = []
        title = True
        avg_metric = {}
        for test_type in mean_score:
            row = [test_type]
            for dataset_name in dataset_names:
                for style in styles:
                    for metric in metrics:
                        if not np.isnan(mean_score[test_type][dataset_name][metric][style]):
                            if title:
                                dataset_row.append(dataset_name)
                                style_row.append(style)
                                metrics_row.append(metric)
                            row.append(mean_score[test_type][dataset_name][metric][style])
                            if metric not in avg_metric:
                                avg_metric[metric] = [mean_score[test_type][dataset_name][metric][style]]
                            else:
                                avg_metric[metric].append(mean_score[test_type][dataset_name][metric][style])
            avg_metric['diversity - vocab'] = [vocab_size[test_type]]
            for m in avg_metric:
                if title:
                    dataset_row.append('total avg.')
                    style_row.append('total avg.')
                    metrics_row.append(m)
                row.append(np.mean(avg_metric[m]))
            total_rows.append(row)
            title = False
        writer.writerow(dataset_row)
        writer.writerow(style_row)
        writer.writerow(metrics_row)
        for r in total_rows:
            writer.writerow(r)
    print(f"finished to write results to {tgt_eval_results_path}")

def main():
    cuda_idx = "3"
    styles = ['factual','positive', 'negative', 'humor', 'romantic']
    data_dir = os.path.join(os.path.expanduser('~'), 'data')
    results_dir = os.path.join(os.path.expanduser('~'), 'results')
    results_evaluation_dir = os.path.join(results_dir, 'evaluation')
    if not os.path.exists(results_evaluation_dir):
        os.makedirs(results_evaluation_dir)
    gt_imgs_for_test = os.path.join(data_dir, 'gt_imgs_for_test')
    #path_test_prompt_manipulation = os.path.join(results_dir,'04_15_54__14_12_2022','results_all_models_source_classes_04_15_54__14_12_2022.csv')
    #path_test_image_manipulation = os.path.join(results_dir,'11_45_38__14_12_2022','results_all_models_source_classes_11_45_38__14_12_2022.csv')
    path_test_prompt_manipulation = os.path.join(results_dir,'evaluation','total_results_prompt_manipulation.csv')
    path_test_image_manipulation = os.path.join(results_dir,'evaluation','total_results_image_manipulation.csv')


    txt_cls_model_paths = {'senticap': os.path.join(os.path.expanduser('~'),'checkpoints','best_model','pos_neg_best_text_style_classification_model.pth'),
                           'flickrstyle10k': os.path.join(os.path.expanduser('~'),'checkpoints','best_model','humor_romantic_best_text_style_classification_model.pth')}

    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    label = cur_time#'25_12_2022_v1' # cur_time
    tgt_eval_results_path = os.path.join(results_evaluation_dir, label+'_eval_results.csv')
    tgt_eval_results_path_for_all_frames = os.path.join(results_evaluation_dir, label+'_eval_results_for_all_frames.csv')
    styles_per_dataset = {'senticap': ['positive', 'negative'],
                           'flickrstyle10k': ['humor', 'romantic']}

    res_paths = {}
    res_paths['prompt_manipulation'] = path_test_prompt_manipulation
    res_paths['image_manipulation'] = path_test_image_manipulation

    dataset_names =['senticap', 'flickrstyle10k']
    metrics = ['bleu', 'rouge', 'CLIPScoreRef','CLIPScore','style_classification', 'fluency']   # ['bleu','rouge','meteor', 'spice', 'CLIPScoreRef','CLIPScore','style_classification', 'fluency']
    metrics = ['style_classification', 'fluency']   # ['bleu','rouge','meteor', 'spice', 'CLIPScoreRef','CLIPScore','style_classification', 'fluency']
    #metrics = ['bleu']   # ['bleu','rouge','meteor', 'spice', 'CLIPScoreRef','CLIPScore','style_classification', 'fluency']
    #metrics = ['bleu','rouge', 'CLIPScoreRef','CLIPScore', 'fluency']   # ['bleu','rouge','meteor', 'spice', 'CLIPScoreRef','CLIPScore','style_classification', 'fluency']
    #metrics = ['fluency']   # ['bleu','rouge','meteor', 'spice', 'CLIPScoreRef','CLIPScore','style_classification', 'fluency']

    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    labels_dict_idxs = {}
    for i, label in enumerate(list(set(list(df_train['category'])))):
        labels_dict_idxs[label] = i

    test_set_path = {}
    txt_cls_model_paths_to_load = {}
    for dataset_name in dataset_names:
        txt_cls_model_paths_to_load[dataset_name] = txt_cls_model_paths[dataset_name]
        test_set_path[dataset_name] = os.path.join(data_dir, dataset_name, 'annotations', 'test.pkl')
    gts_per_data_set = get_gts_data(test_set_path)

    res_data_per_test = get_res_data(res_paths)
    #copy_imgs_to_test_dir(gts_per_data_set, res_data_per_test, styles, metrics, gt_imgs_for_test)
    mean_score, all_scores = calc_score(gts_per_data_set, res_data_per_test, styles, metrics,cuda_idx, data_dir, txt_cls_model_paths_to_load, labels_dict_idxs, gt_imgs_for_test, styles_per_dataset)

    '''
    cider_score = cider(gts, res,styles)
    print(f"cider score  = {cider_score}")
    #meteor_score = meteor(gts, res)
    #print(f"meteor score  = {meteor_score}")
    #spice_score = spice(gts, res)
    '''

    vocab_size = diversitiy(res_data_per_test, gts_per_data_set)

    for test_type in mean_score:
        for dataset in dataset_names:
            for metric in metrics:
                for style in styles:
                    print(f'{test_type}: {dataset} {metric} score for {style} = {mean_score[test_type][dataset][metric][style]}')
    for test_type in res_data_per_test:
        print(f'Vocabulary size for experiment{test_type} dataset is {vocab_size[test_type]}')
    write_results(mean_score, tgt_eval_results_path,dataset_names, metrics, styles, vocab_size)
    write_results_for_all_frames(all_scores, tgt_eval_results_path_for_all_frames, metrics)

    print('Finished to evaluate')

if __name__=='__main__':
    main()