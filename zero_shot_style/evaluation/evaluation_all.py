# try to use it: for adapting to python3
# https: // github.com / sks3i / pycocoevalcap
# https://github.com/wangleihitcs/CaptionMetrics
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate import meteor
# from nltk import word_tokenize
# import os
# import pickle
# import matplotlib.pyplot as plt
import json
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis

import pandas as pds
import shutil
import timeit
from datetime import datetime
from evaluate import load
import math
import pandas as pd
import torch
import csv
import numpy as np
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import os
import pickle
from zero_shot_style.evaluation.pycocoevalcap.bleu.bleu import Bleu
from zero_shot_style.evaluation.pycocoevalcap.cider.cider import Cider
from zero_shot_style.evaluation.pycocoevalcap.meteor.meteor import Meteor
from zero_shot_style.evaluation.pycocoevalcap.rouge.rouge import Rouge
from zero_shot_style.evaluation.pycocoevalcap.spice import Spice
from zero_shot_style.model.ZeroCLIP import CLIPTextGenerator
from zero_shot_style.evaluation.text_style_classification import evaluate as evaluate_text_style_classification
from zero_shot_style.evaluation.text_style_classification import BertClassifier, tokenizer
from argparse import ArgumentParser
from zero_shot_style.utils import get_hparams, replace_user_home_dir

NORMALIZE_GRADE_SCALE = 100
MAX_PERPLEXITY = 500
target_seq_length = 15


class CLIPScoreRef:
    def __init__(self, text_generator):
        self.text_generator = text_generator

    def compute_score(self, gts, res):
        '''

        :param gts: list of str
        :param res: str
        :return:
        '''
        # print("calculate CLIPScoreRef...")
        res_val = res
        if type(res) == dict:
            res_val = list(res.values())[0]
        scores_for_all = []
        for k in gts:
            for gt in gts[k]:
                if '\n' in gt:
                    global_gt = gt
                    global_gt_splitted = global_gt.split('\n')
                    for gt in global_gt_splitted:
                        if len(gt) > 1:
                            if gt[1] == '.':
                                gt = gt[2:]
                        text_features_gt = self.text_generator.get_txt_features(gt, source_clip=True)
                        text_features_ref = self.text_generator.get_txt_features(list(res.values())[0],
                                                                                 source_clip=True)
                        with torch.no_grad():
                            clip_score_ref = (text_features_ref @ text_features_gt.T)
                            score = clip_score_ref.cpu().numpy()
                        scores_for_all.append(score)
                text_features_gt = self.text_generator.get_txt_features(gt, source_clip=True)
                text_features_ref = self.text_generator.get_txt_features(list(res.values())[0], source_clip=True)
                with torch.no_grad():
                    clip_score_ref = (text_features_ref @ text_features_gt.T)
                    score = clip_score_ref.cpu().numpy()
                scores_for_all.append(score)
        avg_score = np.mean(scores_for_all)
        # print('CLIPScoreRef = %s' % avg_score)
        return avg_score, scores_for_all


class CLIPScore:
    def __init__(self, text_generator):
        self.text_generator = text_generator

    def compute_score(self, image_path, res):
        '''

        :param image_path: str full path to the image
        :param res: str
        :return:
        '''
        # print("calculate CLIPScore...")
        res_val = res
        if type(res) == dict:
            res_val = list(res.values())[0]
        image_features = self.text_generator.get_img_feature([image_path], None, source_clip=True)
        text_features = self.text_generator.get_txt_features(res_val, source_clip=True)
        with torch.no_grad():
            clip_score = (image_features @ text_features.T)
        score = clip_score.cpu().numpy()
        # print(f'text: {res}')
        # print('CLIPScore = %s' % score[0][0])
        return score[0][0], [score]


class STYLE_CLS:
    def __init__(self, txt_cls_model_path, data_dir, desired_cuda_num, labels_dict_idxs, hidden_state_to_take=-1,
                 scale_noise=0):
        self.data_dir = data_dir
        self.desired_cuda_num = desired_cuda_num
        self.labels_dict_idxs = labels_dict_idxs
        # self.df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        use_cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda" if use_cuda else "cpu")
        self.hidden_state_to_take = hidden_state_to_take
        self.scale_noise = scale_noise
        self.model = self.load_model(txt_cls_model_path)

    def load_model(self, txt_cls_model_path):
        model = BertClassifier(device=self.device, hidden_state_to_take=self.hidden_state_to_take,
                               scale_noise=self.scale_noise)
        model.to(self.device)
        if 'cuda' in self.device.type:
            checkpoint = torch.load(txt_cls_model_path, map_location='cuda:0')
        else:
            checkpoint = torch.load(txt_cls_model_path, map_location=torch.device(self.device.type))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def compute_score(self, res, gt_label):
        '''

        :param gts: list of text
        :param res: dict. key=str. value=list of single str
        :return:
        '''
        res_val = res
        if type(res) == dict:
            res_val = list(res.values())[0][0]
        res_tokens = tokenizer(res_val, padding='max_length', max_length=512, truncation=True,
                               # todo: check if need to put 40
                               return_tensors="pt")
        # print(f"self.labels_dict_idxs = {self.labels_dict_idxs}")
        # print(f"self.labels_dict_idxs[gt_label] = {self.labels_dict_idxs[gt_label]}")
        gt_label_idx = torch.tensor(self.labels_dict_idxs[gt_label]).to(self.device)
        mask = res_tokens['attention_mask'].to(self.device)
        input_id = res_tokens['input_ids'].squeeze(1).to(self.device)
        output = self.model(input_id, mask)

        outputs_bin = torch.round(torch.tensor([out[0] for out in output])).to(self.device)
        if outputs_bin[0] == gt_label_idx:
            cls_score = 1
        else:
            cls_score = 0

        # normalized_output = output[0]/torch.norm(output[0])

        # cls_score = normalized_output[gt_label_idx]*1+ normalized_output[1-gt_label_idx]*-1

        # cut_values
        # cls_score_np = np.max([0,np.min([cls_score.cpu().data.numpy(),1])])
        return cls_score, None

    def compute_label_for_list(self, res):
        '''

        :param gts: list of labels
        :param res: list of sentences
        :return:
        '''
        res_tokens = tokenizer(res, padding='max_length', max_length=512, truncation=True,
                               # todo check if max need to be 40 like in train
                               return_tensors="pt")
        masks = res_tokens['attention_mask'].to(self.device)
        input_ids = res_tokens['input_ids'].squeeze(1).to(self.device)
        outputs = self.model(input_ids, masks)

        outputs_bin = torch.round(torch.tensor([out[0] for out in outputs])).to(self.device)
        return outputs_bin

    def compute_score_for_total_data(self, gts, res, dataset_name):
        self.df_test = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        total_acc_test_for_all_data = evaluate_text_style_classification(self.model[dataset_name], self.df_test,
                                                                         self.labels_dict_idxs, self.desired_cuda_num)
        return total_acc_test_for_all_data, None


class STYLE_CLS_EMOJI:
    def __init__(self, emoji_vocab_path, maxlen_emoji_sentence, emoji_pretrained_path, idx_emoji_style_dict):
        with open(emoji_vocab_path, 'r') as f:
            self.vocabulary = json.load(f)
        self.emoji_st_tokenizer = SentenceTokenizer(self.vocabulary, maxlen_emoji_sentence)
        self.emoji_style_model = self.load_model(emoji_pretrained_path)
        self.idx_emoji_style_dict = idx_emoji_style_dict

    def load_model(self, emoji_pretrained_path):
        emoji_style_model = torchmoji_emojis(emoji_pretrained_path)
        for param in emoji_style_model.parameters():
            param.requires_grad = False
        emoji_style_model.eval()
        return emoji_style_model

    def compute_score(self, res, gt_label):
        '''
        :param gts: list of text
        :param res: dict. key=str. value=list of single str
        :return:
        '''
        res_val = res
        if type(res) == dict:
            res_val = list(res.values())[0][0]

        with torch.no_grad():
            tokenized, _, _ = self.emoji_st_tokenizer.tokenize_sentences([res_val])
            tokenized = torch.from_numpy(tokenized.astype(np.int32))
            # tokenized = torch.from_numpy(tokenized.astype(np.int32)).to(self.device)
            # self.emoji_style_model.to(torch.device("cuda"))
            # self.emoji_style_model = self.emoji_style_model.to(self.device)

            # print(f"next(self.emoji_style_model.parameters()).is_cuda = {next(self.emoji_style_model.parameters()).is_cuda}")
            # print(f"tokenized.is_cuda={tokenized.is_cuda}")
            emoji_style_probs = torch.tensor(self.emoji_style_model(tokenized))
            cls_score = emoji_style_probs[0,self.idx_emoji_style_dict[gt_label]]
        return cls_score, None

    def compute_label_for_list(self, res):
        '''

        :param gts: list of labels
        :param res: list of sentences
        :return:
        '''
        res_tokens = tokenizer(res, padding='max_length', max_length=512, truncation=True,
                               # todo check if max need to be 40 like in train
                               return_tensors="pt")
        masks = res_tokens['attention_mask'].to(self.device)
        input_ids = res_tokens['input_ids'].squeeze(1).to(self.device)
        outputs = self.model(input_ids, masks)

        outputs_bin = torch.round(torch.tensor([out[0] for out in outputs])).to(self.device)
        return outputs_bin

    def compute_score_for_total_data(self, gts, res, dataset_name):
        self.df_test = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        total_acc_test_for_all_data = evaluate_text_style_classification(self.model[dataset_name], self.df_test,
                                                                         self.labels_dict_idxs, self.desired_cuda_num)
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
        # print(f"data=self.tests:")
        # print(f"{self.tests}")
        # print(f"data=self.tests")
        results = self.perplexity.compute(data=self.tests, model_id=self.model_id,
                                          add_start_token=True)  # check is the source
        # results = self.perplexity.compute(data=self.tests, model_id=self.model_id, add_start_token=True)
        for i, pp in enumerate(results['perplexities']):
            k = self.k[i]
            style = self.style[i]
            metric = self.metric[i]
            score_dict_per_metric[metric][k][style] = 1 - np.min([pp, MAX_PERPLEXITY]) / MAX_PERPLEXITY
            score_per_metric_and_style[metric][style].append(score_dict_per_metric[metric][k][style])
            all_scores = save_all_data_k(all_scores, k, test_type, style, metric, score_dict_per_metric,
                                         res=self.tests[i])  # save all data per key frames
        return score_dict_per_metric, score_per_metric_and_style, all_scores


def save_all_data_k(all_scores, k, test_type, style, metric, score_dict_per_metric, res=None, gts=None,
                    image_path=None):
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
    if image_path and ('image_path' not in all_scores[k][test_type][style]):
        all_scores[k][test_type][style]['image_path'] = image_path
    return all_scores


def evaluate_single_res(res, gt, image_path, label, metrics, evaluation_obj):
    evaluation = {}
    print('evaluate single res.')
    for metric in metrics:
        if metric in ['style_classification', 'style_classification_emoji']:
            if label == 'factual':
                evaluation[metric] = None
                continue
            else:
                evaluation[metric], _ = evaluation_obj[metric].compute_score(res, label)
        elif metric == 'CLIPScore':
            evaluation['CLIPScore'], _ = evaluation_obj[metric].compute_score(image_path, res)
        elif metric == 'fluency':  # calc fluency only on all data
            continue
        else:
            evaluation[metric], _ = evaluation_obj[metric].compute_score(gt, res)
    return evaluation


def calc_score(gts_per_data_set, res, styles, metrics, cuda_idx, data_dir, txt_cls_model_paths_to_load,
               labels_dict_idxs, gt_imgs_for_test, config):
    print("Calculate scores...")
    mean_score = {}
    if ('CLIPScoreRef' in metrics) or ('CLIPScore' in metrics):
        text_generator = CLIPTextGenerator(cuda_idx=cuda_idx, config=config)
    if 'style_classification' in metrics:
        style_cls_obj = STYLE_CLS(txt_cls_model_paths_to_load, data_dir, cuda_idx, labels_dict_idxs,
                                                config['hidden_state_to_take_txt_cls'])
    if 'style_classification_emoji' in config['evaluation_metrics']:
        style_cls_emoji_obj = STYLE_CLS_EMOJI(config['emoji_vocab_path'], config['maxlen_emoji_sentence'],
                                                      config['emoji_pretrained_path'], config['idx_emoji_style_dict'])
    if 'fluency' in metrics:
        fluency_obj = Fluency()
    all_scores = {}
    for test_type in res:
        print(f"Calc scores for experiment: **** {test_type} *****")
        t1_exp = timeit.default_timer();
        mean_score_per_dataset = {}
        score_per_metric_and_style = {}
        score_dict_per_metric = {}
        scores_dict_per_metric = {}
        mean_score_per_metric_and_style = {}
        for metric in metrics:
            print(f"    Calc scores for metric: ***{metric}***")
            t1_metric = timeit.default_timer();
            if metric == 'bleu1':
                scorer = Bleu(n=1)
            if metric == 'bleu3':
                scorer = Bleu(n=3)
            if metric == 'bleu4':
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
            elif metric == 'style_classification_emoji':
                scorer = style_cls_emoji_obj

            score_per_metric_and_style[metric] = {}
            for style in styles:
                score_per_metric_and_style[metric][style] = []
            score_dict_per_metric[metric] = {}
            scores_dict_per_metric[metric] = {}
            mean_score_per_metric_and_style[metric] = {}
            for i1, k in enumerate(res[test_type]):
                if k in gts_per_data_set:
                    score_dict_per_metric[metric][k] = {}
                    scores_dict_per_metric[metric][k] = {}
                    for i2, style in enumerate(styles):
                        if style == 'factual' and metric == 'style_classification':
                            continue
                        if style in gts_per_data_set[k] and style in res[test_type][k]:
                            if not gts_per_data_set[k][style]:
                                continue
                            tmp_res = {k: [res[test_type][k][style]]}
                            if metric == 'CLIPScore':
                                # score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][style] = scorer.compute_score(os.path.join(config['test_imgs'],gts_per_data_set[k]['image_path'].split('/')[-1]), tmp_res)
                                score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][
                                    style] = scorer.compute_score(gts_per_data_set[k]['image_path'], tmp_res)
                                score_per_metric_and_style[metric][style].append(
                                    score_dict_per_metric[metric][k][style])
                                all_scores = save_all_data_k(all_scores, k, test_type, style, metric,
                                                             score_dict_per_metric, res=tmp_res[k][0],
                                                             image_path=gts_per_data_set[k]['image_path'])
                            elif metric in ['style_classification', 'style_classification_emoji']:
                                score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][
                                    style] = scorer.compute_score(tmp_res, style)
                                score_per_metric_and_style[metric][style].append(
                                    score_dict_per_metric[metric][k][style])
                                all_scores = save_all_data_k(all_scores, k, test_type, style, metric,
                                                             score_dict_per_metric, res=tmp_res[k][0])
                            elif metric == 'fluency':
                                if len(list(tmp_res.values())[0][0].split()) < 2:
                                    print(len(list(tmp_res.values())[0][0].split()) < 2)
                                    continue
                                scorer.add_test(tmp_res, metric, k, style)
                            else:
                                tmp_gts = {k: gts_per_data_set[k][style]}
                                score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][
                                    style] = scorer.compute_score(tmp_gts, tmp_res)
                                score_dict_per_metric[metric][k][style] = np.mean(
                                    score_dict_per_metric[metric][k][style])  # todo: check if need it
                                score_per_metric_and_style[metric][style].append(
                                    score_dict_per_metric[metric][k][style])
                                all_scores = save_all_data_k(all_scores, k, test_type, style, metric,
                                                             score_dict_per_metric, res=tmp_res[k][0], gts=tmp_gts[k])
            if metric == 'fluency':
                score_dict_per_metric, score_per_metric_and_style, all_scores = fluency_obj.compute_score(
                    score_dict_per_metric, score_per_metric_and_style, all_scores, test_type)
                # print("score_dict_per_metric:")
                # print(score_dict_per_metric)

            for style in styles:
                if metric == 'fluency':
                    mean_score_per_metric_and_style[metric][style] = np.mean(score_per_metric_and_style[metric][style])
                else:
                    mean_score_per_metric_and_style[metric][style] = np.mean(
                        score_per_metric_and_style[metric][style]) * NORMALIZE_GRADE_SCALE
                print(
                    f"mean_score_per_metric_and_style[metric][{style}] = {mean_score_per_metric_and_style[metric][style]}")
            t2_metric = timeit.default_timer();
            print(
                f"Time to calc this metric: {(t2_metric - t1_metric) / 60} minutes = {t2_metric - t1_metric} seconds.")
        mean_score[test_type] = mean_score_per_metric_and_style
        t2_exp = timeit.default_timer();
        print(f"Time to calc this test: {(t2_exp - t1_exp) / 60} minutes = {t2_exp - t1_exp} seconds.")
    return mean_score, all_scores


def copy_imgs_to_test_dir(gts_per_data_set, res, styles, metrics, gt_imgs_for_test):
    print("Calculate scores...")
    imgs2cpy = []
    for test_type in res:
        for dataset_name in gts_per_data_set:
            for metric in metrics:
                for i1, k in enumerate(res[test_type]):
                    if k in gts_per_data_set:
                        for i2, style in enumerate(styles):
                            if style in gts_per_data_set[k] and style in res[test_type][k]:
                                if not gts_per_data_set[k][style]:
                                    continue
                                if metric == 'CLIPScore':
                                    imgs2cpy.append(gts_per_data_set[k]['image_path'])

    for i in imgs2cpy:
        shutil.copyfile(i, os.path.join(gt_imgs_for_test, i.split('/')[-1]))
    return 0


'''
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
    
def spice(gts, res):
    print("Calculate spice score...")
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('spice = %s' % score)
    return score
'''


def style_accuracy():
    pass


def diversitiy(res, gts):
    print("Calculate vocabulary size...")
    vocab_size = {}
    for test_type in res:
        vocab_size[test_type] = {}
        vocab_list = {}
        for k in res[test_type]:
            for style in res[test_type][k]:
                if style == 'factual':
                    continue
                if style not in vocab_list:
                    vocab_list[style] = []
                tokenized_text = list(map(str.lower, nltk.tokenize.word_tokenize(res[test_type][k][style])))
                vocab_list[style].extend(tokenized_text)
                vocab_list[style] = list(set(vocab_list[style]))
        for style in vocab_list:
            vocab_size[test_type][style] = len(vocab_list[style])
            print(f'Vocabulary size for ***{test_type}, {style}*** is: {vocab_size[test_type][style]}')
    return vocab_size


def get_all_sentences(data_dir, dataset_name, type_set):
    '''
    :param type_set: strind of train/val/test
    :param dataset_name: string with the name of the dataset. 'senticap'/'flickrstyle10k'
    :return: train_sentences: list of all sentences in train set
    '''
    data_path = os.path.join(data_dir, dataset_name, 'annotations', type_set + '.pkl')
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


def get_gts_data(annotations_path, imgs_path, data_type, factual_captions=None, max_num_imgs2test=-1):
    '''
    :param test_set_path: dictionary:keys=dataset names, values=path to pickle file
    :return: gts_per_data_set: key=img_name,values=dict:keys=['image_path','factual','humor','romantic','positive','negative'], values=gt text
    '''
    gts = {}
    with open(os.path.join(annotations_path, data_type + ".pkl"), 'rb') as r:
        data = pickle.load(r)
    for k in data:
        if len(gts) >= max_num_imgs2test and max_num_imgs2test > 0:
            break
        gts[k] = {}
        if factual_captions:
            gts[k]['factual'] = factual_captions[k]
        else:
            gts[k]['factual'] = data[k]['factual']
        gts[k]['image_path'] = replace_user_home_dir(data[k]['image_path'])
        for style in data[k]:
            if style == 'image_path':
                gts[k]['image_path'] = os.path.join(imgs_path, data_type, data[k]['image_path'].split('/')[-1])
                continue
            if style != 'factual':
                gts[k][style] = data[k][style]
    return gts


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
            styles = []
            for row in spamreader:
                if '.jpg' in row[0]:
                    k = row[0].split('.jpg')[0]
                else:
                    k = row[0]
                if 'COCO' in k:
                    k = k.split('_')[-1]
                try:
                    k = int(k)
                except:
                    pass
                if title:
                    styles = row[1:]
                    title = False
                    continue
                else:
                    try:
                        res_data[k] = {}
                        for i, s in enumerate(styles):
                            # res_data[k][s] = row[i+1]
                            # limit sentence to target_seq_length as we create in ZeroStyleCap
                            res_data[k][s] = ' '.join(row[i + 1].split()[:target_seq_length])
                    except:
                        pass
        res_data_per_test[test_type] = res_data
    return res_data_per_test


def write_results_for_all_frames(all_scores, tgt_eval_results_path_for_all_frames, metrics):
    print(f"Write results to {tgt_eval_results_path_for_all_frames}...")
    title = ['k', 'test_type', 'style']
    title.extend(metrics)
    title.extend(['res', 'gts', 'image_path'])
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
                    if 'image_path' in all_scores[k][test_type][style]:
                        row.append(all_scores[k][test_type][style]['image_path'])
                    else:
                        row.append('')
                    writer.writerow(row)
    print(f"finished to write results for all frames in {tgt_eval_results_path_for_all_frames}")


def write_results(mean_score, tgt_eval_results_path, dataset, metrics, styles, vocab_size):
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
            for style in styles:
                for metric in metrics:
                    if not np.isnan(mean_score[test_type][metric][style]):
                        if title:
                            dataset_row.append(dataset)
                            style_row.append(style)
                            metrics_row.append(metric)
                        row.append(mean_score[test_type][metric][style])
                        if metric not in avg_metric:
                            avg_metric[metric] = [mean_score[test_type][metric][style]]
                        else:
                            avg_metric[metric].append(mean_score[test_type][metric][style])
                if 'diversity-vocab_size' not in avg_metric:
                    avg_metric['diversity-vocab_size'] = [vocab_size[test_type][style]]
                else:
                    if style in vocab_size[test_type]:
                        avg_metric['diversity-vocab_size'].append(vocab_size[test_type][style])
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


def get_all_paths_of_tests_txt_style(factual_wo_prompt):
    def add_suffix_to_file_name(files_list):
        fixed_file_names = []
        for f in files_list:
            fixed_file_names.append(f.split('.csv')[0] + '_factual_wo_prompt.csv')
        return fixed_file_names

    # todo:
    base_path = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/20_2_23/')
    # text style
    src_dir_text_style = os.path.join(base_path, 'text_style')
    src_dir_text_style = base_path
    text_style_dir_path = os.listdir(src_dir_text_style)
    tgt_path_text_style = os.path.join(src_dir_text_style, 'total_results_text_style.csv')
    return tgt_path_text_style


def get_all_paths_of_tests():
    # todo:
    base_path = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/12_2_23/')
    # base_path = os.path.join(os.path.expanduser('~'),'experiments/stylized_zero_cap_experiments/20_2_23/')
    # prompt_manipulation
    # src_dir_prompt_manipulation = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/7_2_23/prompt_manipulation')
    src_dir_prompt_manipulation = os.path.join(base_path, 'prompt_manipulation')
    prompt_manipulation_dir_path = os.listdir(src_dir_prompt_manipulation)
    tgt_path_prompt_manipulation = os.path.join(src_dir_prompt_manipulation, 'total_results_prompt_manipulation.csv')

    # image and prompt_manipulation
    # src_dir_image_and_prompt_manipulation = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/7_2_23/image_and_prompt_manipulation')
    src_dir_image_and_prompt_manipulation = os.path.join(base_path, 'image_and_prompt_manipulation')
    image_and_prompt_manipulation_dir_path = os.listdir(src_dir_image_and_prompt_manipulation)
    tgt_path_image_and_prompt_manipulation = os.path.join(src_dir_image_and_prompt_manipulation,
                                                          'total_results_image_and_prompt_manipulation.csv')

    # text style
    # src_dir_text_style = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/7_2_23/text_style')
    src_dir_text_style = os.path.join(base_path, 'text_style')
    text_style_dir_path = os.listdir(src_dir_text_style)
    tgt_path_text_style = os.path.join(src_dir_text_style, 'total_results_text_style.csv')

    # image manipulation
    # src_dir_image_manipulation = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/7_2_23/image_manipulation')
    src_dir_image_manipulation = os.path.join(base_path, 'image_manipulation')
    image_manipulation_dir_path = os.listdir(src_dir_image_manipulation)
    tgt_path_image_manipulation = os.path.join(src_dir_image_manipulation, 'total_results_image_manipulation.csv')
    return tgt_path_prompt_manipulation, tgt_path_image_manipulation, tgt_path_image_and_prompt_manipulation, tgt_path_text_style


def get_all_paths_of_tests_ZeroStyleCap(factual_wo_prompt):
    def add_suffix_to_file_name(files_list):
        fixed_file_names = []
        for f in files_list:
            fixed_file_names.append(f.split('.csv')[0] + '_factual_wo_prompt.csv')
        return fixed_file_names

    # todo:
    base_path = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/12_2_23/')
    base_path = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/20_2_23/')
    base_path = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/23_2_23/')
    # ZeroStyleCap8
    src_dir_ZeroStyleCap8 = os.path.join(base_path, 'ZeroStyleCap8')
    tgt_path_ZeroStyleCap8 = os.path.join(src_dir_ZeroStyleCap8, 'total_results_ZeroStyleCap8.csv')

    # ZeroStyleCap39
    src_dir_ZeroStyleCap39 = os.path.join(base_path, 'ZeroStyleCap39')
    tgt_path_ZeroStyleCap39 = os.path.join(src_dir_ZeroStyleCap39, 'total_results_ZeroStyleCap39.csv')

    # ZeroStyleCapPast
    src_dir_ZeroStyleCapPast = os.path.join(base_path, 'ZeroStyleCapPast')
    tgt_path_ZeroStyleCapPast = os.path.join(src_dir_ZeroStyleCapPast, 'total_results_ZeroStyleCapPast.csv')

    # text style
    # src_dir_text_style = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/7_2_23/text_style')
    text_style_base_path = os.path.join(os.path.expanduser('~'), 'experiments/stylized_zero_cap_experiments/12_2_23/')
    src_dir_text_style = os.path.join(text_style_base_path, 'text_style')
    tgt_path_text_style = os.path.join(src_dir_text_style, 'total_results_text_style.csv')

    if factual_wo_prompt:
        files_list = [tgt_path_ZeroStyleCap8, tgt_path_ZeroStyleCap39, tgt_path_ZeroStyleCapPast]
        tgt_path_ZeroStyleCap8, tgt_path_ZeroStyleCap39, tgt_path_ZeroStyleCapPast = add_suffix_to_file_name(files_list)

    return tgt_path_ZeroStyleCap8, tgt_path_ZeroStyleCap39, tgt_path_ZeroStyleCapPast, tgt_path_text_style


# def analyze_fluency(all_scores,config):
#     ############## histogram
#     fluency_statistic = {}
#     for k in all_scores:
#         for test_type in all_scores[k]:
#             if test_type not in fluency_statistic:
#                 fluency_statistic[test_type] = {}
#             for style in all_scores[k][test_type]:
#                 if style not in fluency_statistic[test_type]:
#                     fluency_statistic[test_type][style] = [all_scores[k][test_type][style]['fluency']]
#                 else:
#                     fluency_statistic[test_type][style].append(all_scores[k][test_type][style]['fluency'])
#
#     print('finish to calc statistic of fluency')
#     # Generate data on commute times.
#     size, scale = 1000, 10
#     commutes = pd.Series(np.random.gamma(scale, size=size) ** 1.5)
#     commutes = pd.Series([1,1,2,2,3,3,3,3,3])
#     fig1 = plt.gcf()
#     for i, test_type in enumerate(fluency_statistic):
#         for j, style in enumerate(fluency_statistic[test_type]):
#             plt.subplot(len(fluency_statistic),len(fluency_statistic[test_type]),i*len(fluency_statistic[test_type])+j+1)
#
#             commutes = pd.Series(fluency_statistic[test_type][style])
#             commutes.plot.hist(grid=True, bins=20, rwidth=0.9,
#                                color='#607c8e')
#             plt.title(f'Fluency Histogram for: {test_type} - {style}')
#             plt.xlabel('Counts')
#             plt.ylabel('Commute Time')
#             plt.grid(axis='y', alpha=0.75)
#     plt.show(block=False)
#     os.makedirs(config['dir_path_for_eval_only_fluency'], exist_ok=True)
#     fig1.savefig(os.path.join(config['dir_path_for_eval_only_fluency'], f'_{test_type}_{style}.png'))
#     ##############

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default=os.path.join('.', 'configs', 'evaluation_config.yaml'),
                        help='full path to config file')
    parser.add_argument("--cuda_idx", type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    config = get_hparams(args)
    data_dir = os.path.join(os.path.expanduser('~'), 'data')
    results_dir = os.path.join(os.path.expanduser('~'), 'results')
    results_evaluation_dir = os.path.join(results_dir, 'evaluation')
    os.makedirs(results_evaluation_dir, exist_ok=True)
    gt_imgs_for_test = os.path.join(data_dir, 'gt_imgs_for_test')

    if config['dataset'] == 'senticap':
        with open(config['factual_captions_path'], 'rb') as f:
            factual_captions = pickle.load(f)
    else:
        factual_captions = None
    gts_per_data_set = get_gts_data(config['annotations_path'], config['imgs_path'], config['data_type'],
                                    factual_captions, config['max_num_imgs2test'])

    res_data_per_test = get_res_data(config['res_path2eval'])
    # copy_imgs_to_test_dir(gts_per_data_set, res_data_per_test, styles, metrics, gt_imgs_for_test)
    # exit(0)
    mean_score, all_scores = calc_score(gts_per_data_set, res_data_per_test, config['styles'], config['metrics'],
                                        config['cuda_idx'], data_dir, config['txt_cls_model_paths'],
                                        config['labels_dict_idxs'], gt_imgs_for_test, config)

    vocab_size = diversitiy(res_data_per_test, gts_per_data_set)
    # analyze_fluency(all_scores,config)

    for test_type in res_data_per_test:
        for metric in config['metrics']:
            for style in config['styles']:
                print(
                    f"{test_type}: {config['dataset']}: {metric} score for {style} = {mean_score[test_type][metric][style]}")
    for test_type in res_data_per_test:
        print(f'Vocabulary size for experiment {test_type} dataset is {vocab_size[test_type]}')

    tgt_eval_results_file_name = os.path.join(list(config['res_path2eval'].values())[0].rsplit('/', 1)[0],
                                              config['tgt_eval_results_file_name'])
    tgt_eval_results_file_name_for_all_frames = os.path.join(
        list(config['res_path2eval'].values())[0].rsplit('/', 1)[0],
        config['tgt_eval_results_file_name_for_all_frames'])

    write_results(mean_score, tgt_eval_results_file_name, config['dataset'], config['metrics'], config['styles'],
                  vocab_size)
    write_results_for_all_frames(all_scores, tgt_eval_results_file_name_for_all_frames, config['metrics'])

    print('Finished to evaluate')


if __name__ == '__main__':
    main()
