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
from torch import nn

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

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# from zero_shot_style.evaluation.pycocoevalcap.bleu.bleu import Bleu
# from zero_shot_style.evaluation.pycocoevalcap.cider.cider import Cider
# from zero_shot_style.evaluation.pycocoevalcap.meteor.meteor import Meteor
# from zero_shot_style.evaluation.pycocoevalcap.rouge.rouge import Rouge
# from zero_shot_style.evaluation.pycocoevalcap.spice import Spice
from zero_shot_style.model.ZeroCLIP import CLIPTextGenerator
from zero_shot_style.evaluation.text_style_classification import evaluate as evaluate_text_style_classification
from zero_shot_style.evaluation.text_style_classification import BertClassifier, tokenizer
from argparse import ArgumentParser

from zero_shot_style.utils import get_hparams, replace_user_home_dir

NORMALIZE_GRADE_SCALE = 100
MAX_PERPLEXITY = 1500
target_seq_length = 30


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
            res_val = [list(res.values())[0][0][:77]]
        image_features = self.text_generator.get_img_feature([image_path], None, source_clip=True)
        text_features = self.text_generator.get_txt_features(res_val, source_clip=True)
        with torch.no_grad():
            clip_score = (image_features @ text_features.T)
        score = clip_score.cpu().numpy()
        # print(f'text: {res}')
        # print('CLIPScore = %s' % score[0][0])
        return score[0][0], [score]

class STYLE_CLS_ROBERTA:
    def __init__(self, finetuned_roberta_config,finetuned_roberta_model_path, desired_cuda_num, labels_dict_idxs_roberta, data_dir=None, max_batch_size=100):
        self.data_dir = data_dir
        self.desired_cuda_num = desired_cuda_num
        self.labels_dict_idxs_roberta = labels_dict_idxs_roberta
        # self.df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        use_cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda" if use_cuda else "cpu")
        # self.sentiment_model = self.load_model(finetuned_roberta_config,finetuned_roberta_model_path)
        self.max_batch_size = max_batch_size

        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        self.sentiment_model.to(self.device)
        self.sentiment_model.eval()
        # SENTIMENT: Freeze sentiment model weights
        for param in self.sentiment_model.parameters():
            param.requires_grad = False
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # # SENTIMENT: tokenizer for sentiment analysis module
        # task = 'sentiment'
        # base_roberta_model = f"cardiffnlp/twitter-roberta-base-{task}-latest"
        # self.sentiment_tokenizer = AutoTokenizer.from_pretrained(base_roberta_model)
        # # SENTIMENT: fields for type and scale of sentiment
        # self.sentiment_scale = 1

    def preprocess(self, text):
        def preprocess_single_text(text):
            new_text = []
            for t in text.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t
                new_text.append(t)
            return " ".join(new_text)

        if type(text) == list:
            new_text_list = []
            for t in text:
                new_text_list.append(preprocess_single_text(t))
            return new_text_list
        else:
            return preprocess_single_text(text)

    def load_model(self, finetuned_roberta_config,finetuned_roberta_model_path):
        f_roberta_config = AutoConfig.from_pretrained(finetuned_roberta_config)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            finetuned_roberta_model_path,
            config=f_roberta_config)
        sentiment_model.to(self.device)
        sentiment_model.eval()

        # SENTIMENT: Freeze sentiment model weights
        for param in sentiment_model.parameters():
            param.requires_grad = False

        return sentiment_model

    def compute_score(self, res, gt_label):
        '''

        :param gts: list of text
        :param res: dict. key=str. value=list of single str
        :return:
        '''
        res_val = res
        if type(res) == dict:
            res_val = list(res.values())[0][0]
        text = self.preprocess(res_val)
        with torch.no_grad():
            # inputs = self.sentiment_tokenizer([text], padding=True, return_tensors="pt")
            encoded_input = self.sentiment_tokenizer(text, return_tensors="pt").to(self.device)
            output = self.sentiment_model(**encoded_input)
            scores = output[0][0].detach()
            scores = nn.functional.softmax(scores)

            # inputs['input_ids'] = inputs['input_ids'].to(self.sentiment_model.device)
            # inputs['attention_mask'] = inputs['attention_mask'].to(self.sentiment_model.device)
            # logits = self.sentiment_model(**inputs)['logits']
            # output = nn.functional.softmax(logits[0], dim=-1)  # todo:check it

            # relevant_logits = torch.tensor([logits[:,i] for i in [0,2]])
            # output = nn.functional.softmax(relevant_logits, dim=-1) #todo:check it
            cls_score = scores[self.labels_dict_idxs_roberta[gt_label]].item()
        return cls_score, None

    def compute_label_for_list(self, res, gt_label):
        '''

        :param gts: list of labels
        :param res: list of sentences
        :return:
        '''
        # text = self.preprocess(res)
        # with torch.no_grad():
        #     # inputs = self.sentiment_tokenizer([text], padding=True, return_tensors="pt")
        #     encoded_input = self.sentiment_tokenizer(text, return_tensors="pt")
        #     output = self.sentiment_model(**encoded_input)
        #
        with torch.no_grad():
            text_list = self.preprocess(res)
            encoded_input = self.sentiment_tokenizer(text_list, padding=True, return_tensors='pt').to(self.device)
            output = self.sentiment_model(**encoded_input)
            scores = output[0].detach()
            scores1 = nn.functional.softmax(scores, dim=-1)
            cls_scores = scores1[:,self.labels_dict_idxs_roberta[gt_label]]
        return cls_scores

        # with torch.no_grad():
        #     # if type(gt_label_idx)==type('str'):
        #     #     gt_label_idx = self.labels_dict_idxs[gt_label_idx]
        #     total_outputs = torch.tensor([]).to(self.device)
        #     for i in range(round(np.ceil(len(res)/self.max_batch_size))):
        #         part_res = res[i*self.max_batch_size:(i+1)*self.max_batch_size]
        #         inputs = self.sentiment_tokenizer(part_res, padding=True, return_tensors="pt")
        #         inputs['input_ids'] = inputs['input_ids'].to(self.sentiment_model.device)
        #         inputs['attention_mask'] = inputs['attention_mask'].to(self.sentiment_model.device)
        #         logits = self.sentiment_model(**inputs)['logits']
        #         relevant_logits = [logits[:, i] for i in [0, 2]] # todo:check it
        #         output = nn.functional.softmax(relevant_logits, dim=-1)  # todo:check it
        #         cls_scores = torch.sum(output[:,self.labels_dict_idxs_roberta[gt_label]])#todo
        #     return cls_scores

    def compute_score_for_total_data(self, gts, res, dataset_name):
        self.df_test = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        total_acc_test_for_all_data = evaluate_text_style_classification(self.model[dataset_name], self.df_test,
                                                                         self.labels_dict_idxs, self.desired_cuda_num)
        return total_acc_test_for_all_data, None




class STYLE_CLS:
    def __init__(self, txt_cls_model_path, desired_cuda_num, labels_dict_idxs, data_dir=None, hidden_state_to_take=-1,
                 scale_noise=0, max_batch_size=100):
        self.data_dir = data_dir
        self.desired_cuda_num = desired_cuda_num
        self.labels_dict_idxs = labels_dict_idxs
        # self.df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        use_cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda" if use_cuda else "cpu")
        self.hidden_state_to_take = hidden_state_to_take
        self.scale_noise = scale_noise
        self.tokenizer = tokenizer
        self.model = self.load_model(txt_cls_model_path)
        self.max_batch_size = max_batch_size

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

        ### binary output
        outputs_bin = torch.round(torch.tensor([out[0] for out in output])).to(self.device)
        if outputs_bin[0] == gt_label_idx:
            cls_score = 1
        else:
            cls_score = 0

        ##############
        # # continuous output
        # if gt_label_idx == 1:
        #     cls_score = torch.tensor([out[0] for out in output]).to(self.device)
        # else:
        #     cls_score = torch.tensor([1-out[0] for out in output]).to(self.device)
        # ##############
        # normalized_output = output[0]/torch.norm(output[0])

        # cls_score = normalized_output[gt_label_idx]*1+ normalized_output[1-gt_label_idx]*-1

        # cut_values
        # cls_score_np = np.max([0,np.min([cls_score.cpu().data.numpy(),1])])
        return cls_score, None

    def compute_label_for_list(self, res, gt_label_idx):
        '''

        :param gts: list of labels
        :param res: list of sentences
        :return:
        '''
        with torch.no_grad():
            if type(gt_label_idx)==type('str'):
                gt_label_idx = self.labels_dict_idxs[gt_label_idx]
            total_outputs = torch.tensor([]).to(self.device)
            for i in range(round(np.ceil(len(res)/self.max_batch_size))):
                part_res = res[i*self.max_batch_size:(i+1)*self.max_batch_size]
                res_tokens = tokenizer(part_res, padding='max_length', max_length=512, truncation=True,
                                       # todo check if max need to be 40 like in train
                                       return_tensors="pt")
                masks = res_tokens['attention_mask'].to(self.device)
                input_ids = res_tokens['input_ids'].squeeze(1).to(self.device)
                outputs = self.model(input_ids, masks)
                total_outputs = torch.cat((total_outputs, outputs))
            ## binary output
            outputs_bin = torch.round(torch.tensor([out[0] for out in total_outputs])).to(self.device)
            cls_scores = [1 if single_out == gt_label_idx else 0 for single_out in outputs_bin]
            ## continuous output
            # if gt_label_idx == 1:
            #     cls_scores = torch.tensor([out[0] for out in outputs]).to(self.device)
            # else:
            #     cls_scores = torch.tensor([1-out[0] for out in outputs]).to(self.device)
        return cls_scores

    def compute_score_for_total_data(self, gts, res, dataset_name):
        self.df_test = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        total_acc_test_for_all_data = evaluate_text_style_classification(self.model[dataset_name], self.df_test,
                                                                         self.labels_dict_idxs, self.desired_cuda_num)
        return total_acc_test_for_all_data, None


class STYLE_CLS_EMOJI:
    def __init__(self, emoji_vocab_path, maxlen_emoji_sentence, emoji_pretrained_path, idx_emoji_style_dict, use_single_emoji_style, desired_labels):
        with open(emoji_vocab_path, 'r') as f:
            self.vocabulary = json.load(f)
        self.emoji_st_tokenizer = SentenceTokenizer(self.vocabulary, maxlen_emoji_sentence)
        self.emoji_style_model = self.load_model(emoji_pretrained_path)
        self.idx_emoji_style_dict = idx_emoji_style_dict
        self.use_single_emoji_style = use_single_emoji_style
        self.desired_labels = desired_labels



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
            # cls_score = emoji_style_probs[0,self.idx_emoji_style_dict[gt_label]]



            ##############
            #suppose there is only one example
            emoji_style_grades = emoji_style_probs[:, self.idx_emoji_style_dict[gt_label]].sum(-1)
            cls_score = emoji_style_grades
            # emoji_style_grades_normalized = emoji_style_grades / torch.sum(emoji_style_grades) # for several examples
            # ##############
            # if self.use_single_emoji_style:
            #     desired_labels_idxs = []
            #     for label in self.desired_labels:
            #         desired_labels_idxs.append(self.idx_emoji_style_dict[label])
            #     emoji_style_probs = emoji_style_probs[:, desired_labels_idxs]
            #     # normalize each row sample
            #     emoji_style_probs = emoji_style_probs / torch.unsqueeze(torch.sum(emoji_style_probs, dim=-1), 1)
            #     cls_score = emoji_style_probs[:,torch.tensor(self.desired_labels.index(label))]
            # else: #use several emojis
            #     cls_score = sum(emoji_style_probs[:, self.idx_emoji_style_dict[gt_label]])
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

    def compute_score(self, score_dict_per_metric, score_per_metric_and_style, all_scores, test_name):
        '''
        sentence = list(res.values())[0]
        results = self.perplexity.compute(data=sentence, model_id=self.model_id, add_start_token=False)
        return results['mean_perplexity'], results['perplexities']
        '''
        # print(f"data=self.tests:")
        # print(f"{self.tests}")
        # print(f"data=self.tests")
        # self.tests = ['Despite the efforts, the final finish of the ski touring system ended in disappointment.',
        #                 'Regrettably, the inefficient sucksling board is utilized for extracting water from educational materials.',
        #                 'Winter hiking in the Canadian wilderness can be extremely challenging, as depicted in this evocative photograph by The Man.']
        # results = f.perplexity.compute(data=["A wonderful banner parachute-fold over double."], model_id=self.model_id,
        #                                   add_start_token=True)  # check is the source

        results = self.perplexity.compute(data=self.tests, model_id=self.model_id,
                                          add_start_token=True)  # check is the source
        # results = self.perplexity.compute(data=self.tests, model_id=self.model_id, add_start_token=True)
        for i, pp in enumerate(results['perplexities']):
            k = self.k[i]
            style = self.style[i]
            metric = self.metric[i]
            # print(1 - np.min([pp, MAX_PERPLEXITY]) / MAX_PERPLEXITY)
            score_dict_per_metric[metric][k][style] = 1 - np.min([pp, MAX_PERPLEXITY]) / MAX_PERPLEXITY
            score_per_metric_and_style[metric][style].append(score_dict_per_metric[metric][k][style])
            all_scores = save_all_data_k(all_scores, k, test_name, style, metric, score_dict_per_metric,
                                         res=self.tests[i])  # save all data per key frames
        return score_dict_per_metric, score_per_metric_and_style, all_scores


def save_all_data_k(all_scores, k, test_name, style, metric, score_dict_per_metric, res=None, gts=None,
                    image_path=None):
    # save all data per key frames
    if k not in all_scores:
        all_scores[k] = {}
    if test_name not in all_scores[k]:
        all_scores[k][test_name] = {}
    if style not in all_scores[k][test_name]:
        all_scores[k][test_name][style] = {}

    all_scores[k][test_name][style][metric] = score_dict_per_metric[metric][k][style]

    if res and ('res' not in all_scores[k][test_name][style]):
        all_scores[k][test_name][style]['res'] = res
    if gts and ('gts' not in all_scores[k][test_name][style]):
        all_scores[k][test_name][style]['gts'] = gts
    if image_path and ('image_path' not in all_scores[k][test_name][style]):
        all_scores[k][test_name][style]['image_path'] = image_path
    return all_scores


def evaluate_single_res(res, gt, image_path, label, metrics, evaluation_obj):
    evaluation = {}
    print('evaluate single res.')
    for metric in metrics:
        if metric in ['style_classification', 'style_classification_emoji', 'style_classification_roberta']:
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
    print("Calculate scores:")
    mean_score = {}
    std_score = {}
    median_score = {}
    if ('CLIPScoreRef' in metrics) or ('CLIPScore' in metrics):
        text_generator = CLIPTextGenerator(cuda_idx=cuda_idx, config=config)
    if 'style_classification' in metrics:
        style_cls_obj = STYLE_CLS(txt_cls_model_paths_to_load, cuda_idx, labels_dict_idxs, data_dir,
                                                config['hidden_state_to_take_txt_cls'])
        print(f"style_cls_obj = STYLE_CLS")
    if 'style_classification_emoji' in config['metrics']:
        style_cls_emoji_obj = STYLE_CLS_EMOJI(config['emoji_vocab_path'], config['maxlen_emoji_sentence'],
                                                      config['emoji_pretrained_path'], config['idx_emoji_style_dict'], config['use_single_emoji_style'], config['desired_labels'])
    if 'style_classification_roberta' in config['metrics']:
        style_cls_obj = STYLE_CLS_ROBERTA(config['finetuned_roberta_config'],config['finetuned_roberta_model_path'], cuda_idx, config['labels_dict_idxs_roberta'], data_dir)
        print(f"style_cls_obj = STYLE_CLS_ROBERTA")


    if 'fluency' in metrics:
        fluency_obj = Fluency()
    all_scores = {}
    #go over all experiements
    for test_name in res:
        print(f"Calc scores for experiment: **** {test_name} *****")
        t1_exp = timeit.default_timer();
        mean_score_per_dataset = {}
        score_per_metric_and_style = {}
        score_dict_per_metric = {}
        scores_dict_per_metric = {}
        mean_score_per_metric_and_style = {}
        std_score_per_metric_and_style = {}
        median_score_per_metric_and_style = {}
        # initial evaluation objects
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
            elif metric == 'style_classification_roberta':
                scorer = style_cls_obj
            elif metric == 'style_classification_emoji':
                scorer = style_cls_emoji_obj

            score_per_metric_and_style[metric] = {}
            for style in styles:
                score_per_metric_and_style[metric][style] = []
            score_dict_per_metric[metric] = {}
            scores_dict_per_metric[metric] = {}
            mean_score_per_metric_and_style[metric] = {}
            std_score_per_metric_and_style[metric] = {}
            median_score_per_metric_and_style[metric] = {}
            # go over each image caption from res
            for i1, k in enumerate(res[test_name]):
                if k not in gts_per_data_set:
                    continue
                if True:
                # if k in gts_per_data_set:
                    score_dict_per_metric[metric][k] = {}
                    scores_dict_per_metric[metric][k] = {}
                    for i2, style in enumerate(styles):
                        if style == 'factual' and (metric == 'style_classification' or metric == 'style_classification_roberta'):
                            continue
                        # if style in gts_per_data_set[k] and style in res[test_name][k]:
                        if style in res[test_name][k]:
                            # if not gts_per_data_set[k][style]:
                            #     continue
                            tmp_res = {k: [res[test_name][k][style]]}
                            if metric == 'CLIPScore':
                                # score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][style] = scorer.compute_score(os.path.join(config['test_imgs'],gts_per_data_set[k]['image_path'].split('/')[-1]), tmp_res)
                                score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][
                                    style] = scorer.compute_score(gts_per_data_set[k]['image_path'], tmp_res)
                                score_per_metric_and_style[metric][style].append(
                                    score_dict_per_metric[metric][k][style])
                                all_scores = save_all_data_k(all_scores, k, test_name, style, metric,
                                                             score_dict_per_metric, res=tmp_res[k][0],
                                                             image_path=gts_per_data_set[k]['image_path'])
                            elif metric in ['style_classification','style_classification_roberta', 'style_classification_emoji']:
                                score_dict_per_metric[metric][k][style], scores_dict_per_metric[metric][k][
                                    style] = scorer.compute_score(tmp_res, style)
                                score_per_metric_and_style[metric][style].append(
                                    score_dict_per_metric[metric][k][style])
                                all_scores = save_all_data_k(all_scores, k, test_name, style, metric,
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
                                all_scores = save_all_data_k(all_scores, k, test_name, style, metric,
                                                             score_dict_per_metric, res=tmp_res[k][0], gts=tmp_gts[k])
            if metric == 'fluency':
                score_dict_per_metric, score_per_metric_and_style, all_scores = fluency_obj.compute_score(
                    score_dict_per_metric, score_per_metric_and_style, all_scores, test_name)
                # print("score_dict_per_metric:")
                # print(score_dict_per_metric)

            for style in styles:
                if metric == 'fluency':
                    mean_score_per_metric_and_style[metric][style] = np.mean(score_per_metric_and_style[metric][style])
                    std_score_per_metric_and_style[metric][style] = np.std(score_per_metric_and_style[metric][style])
                    median_score_per_metric_and_style[metric][style] = np.median(score_per_metric_and_style[metric][style])
                else:
                    try:
                        mean_score_per_metric_and_style[metric][style] = np.mean(
                        score_per_metric_and_style[metric][style]) * NORMALIZE_GRADE_SCALE
                        std_score_per_metric_and_style[metric][style] = np.std(
                        score_per_metric_and_style[metric][style]) * NORMALIZE_GRADE_SCALE
                        median_score_per_metric_and_style[metric][style] = np.median(
                        score_per_metric_and_style[metric][style]) * NORMALIZE_GRADE_SCALE

                    except:
                        mean_score_per_metric_and_style[metric][style] = np.mean([s[0].cpu() for s in
                            score_per_metric_and_style[metric][style]]) * NORMALIZE_GRADE_SCALE
                        std_score_per_metric_and_style[metric][style] = np.std([s[0].cpu() for s in
                                                                                  score_per_metric_and_style[metric][
                                                                                      style]]) * NORMALIZE_GRADE_SCALE
                        median_score_per_metric_and_style[metric][style] = np.median([s[0].cpu() for s in
                                                                                  score_per_metric_and_style[metric][
                                                                                      style]]) * NORMALIZE_GRADE_SCALE
                print(
                    f"mean_score_per_metric_and_style[metric][{style}] = {mean_score_per_metric_and_style[metric][style]},\
                    std_score_per_metric_and_style[metric][{style}] = {std_score_per_metric_and_style[metric][style]},\
                    median_score_per_metric_and_style[metric][{style}] = {median_score_per_metric_and_style[metric][style]}")
            t2_metric = timeit.default_timer();
            print(
                f"Time to calc this metric: {(t2_metric - t1_metric) / 60} minutes = {t2_metric - t1_metric} seconds.")
        mean_score[test_name] = mean_score_per_metric_and_style
        std_score[test_name] = std_score_per_metric_and_style
        median_score[test_name] = median_score_per_metric_and_style
        t2_exp = timeit.default_timer();
        print(f"Time to calc this test: {(t2_exp - t1_exp) / 60} minutes = {t2_exp - t1_exp} seconds.")
    return mean_score, all_scores, std_score, median_score


def copy_imgs_to_test_dir(gts_per_data_set, res, styles, metrics, gt_imgs_for_test):
    print("Calculate scores...")
    imgs2cpy = []
    for test_name in res:
        for dataset_name in gts_per_data_set:
            for metric in metrics:
                for i1, k in enumerate(res[test_name]):
                    if k in gts_per_data_set:
                        for i2, style in enumerate(styles):
                            if style in gts_per_data_set[k] and style in res[test_name][k]:
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
    for test_name in res:
        vocab_size[test_name] = {}
        vocab_list = {}
        for k in res[test_name]:
            for style in res[test_name][k]:
                # if style == 'factual':
                #     continue
                if style not in vocab_list:
                    vocab_list[style] = []
                tokenized_text = list(map(str.lower, nltk.tokenize.word_tokenize(res[test_name][k][style])))
                vocab_list[style].extend(tokenized_text)
                vocab_list[style] = list(set(vocab_list[style]))
        for style in vocab_list:
            vocab_size[test_name][style] = len(vocab_list[style])
            print(f'Vocabulary size for ***{test_name}, {style}*** is: {vocab_size[test_name][style]}')
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


def get_gts_data(annotations_path, imgs_path, data_split, factual_captions=None, max_num_imgs2test=-1):
    '''
    :param annotations_path: dictionary:keys=dataset names, values=path to pickle file
    factual_captions: need to b none for flickrstyle10k
    :return: gts_per_data_set: key=img_name,values=dict:keys=['image_path','factual','humor','romantic','positive','negative'], values=gt text
    '''
    gts = {}
    with open(os.path.join(annotations_path,data_split+'.pkl'), 'rb') as r:
        data = pickle.load(r)
    # import random
    # print(random.sample(list(data.keys()), 20))
    for k in data:
        gts[k] = {}
        if factual_captions:
            gts[k]['factual'] = factual_captions[k]
        else:
            gts[k]['factual'] = data[k]['factual']
        gts[k]['image_path'] = replace_user_home_dir(data[k]['image_path'])
        for style in data[k]:
            if style == 'image_path':
                gts[k]['image_path'] = os.path.join(imgs_path, data_split, data[k]['image_path'].split('/')[-1])
                continue
            if style != 'factual':
                gts[k][style] = data[k][style]
    return gts



def get_res_data_GPT(res_paths):
    '''

    :param res_paths: dict. keys:  'prompt_manipulation', 'image_manipulation'. values: path to res
    :return: res_data_per_test: dict. keys:  'prompt_manipulation', 'image_manipulation'. values: dict to res per image name and style
    '''
    res_data_per_test_source = {}
    res_data_per_test_gpt = {}
    i = 1 #0 - idx_img_name_in_res
    j = 2 # caption res in column idx=2
    idxs_source = []
    idxs_gpt = []
    for test_name in res_paths:
        if test_name.startswith('capdec'):
            i = 0
            j = 1
        res_data_source = {}
        res_data_gpt = {}
        with open(res_paths[test_name], 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            title = True
            styles = []
            for row in spamreader:
                if row[1]=='':
                    continue
                if row[0]=='gpt':
                    s_row = row.copy()
                    row[1] = s_row[1].split(' ')[0]
                    row[2] = s_row[1].split(' - ')[1]
                if '.jpg' in row[i]:
                    k = row[i].split('.jpg')[0]
                else:
                    k = row[i]
                if 'COCO' in k:
                    k = k.split('_')[-1]
                try:
                    k = int(k)  
                except:
                    pass
                if title:
                    if row[1]=='factual':
                        i = 0
                        j = 1
                    try:
                        i = row.index('img_num')
                        j = row.index('img_num') + 1
                    except:
                        pass
                    styles = row[i+1:]
                    styles.remove('')
                    title = False
                    continue
                else:
                    try:
                        if row[0] == 'source':
                            idxs_source.append(k)
                            res_data_source[k] = {}
                            for i_s, s in enumerate(styles):
                                # res_data[k][s] = row[i+1]
                                # limit sentence to target_seq_length as we create in ZeroStyleCap
                                res_data_source[k][s] = ' '.join(row[i_s + j].split()[:target_seq_length])
                        elif row[0] == 'gpt':
                            res_data_gpt[k] = {}
                            idxs_gpt.append(k)
                            for i_s, s in enumerate(styles):
                                # res_data[k][s] = row[i+1]
                                # limit sentence to target_seq_length as we create in ZeroStyleCap
                                if row[i_s + j]=="":
                                    continue
                                res_data_gpt[k][s] = ' '.join(row[i_s + j].split()[:target_seq_length])
                    except:
                        pass

                if row[0] == 'source':
                    res_data_per_test_source[test_name] = res_data_source
                elif row[0] == 'gpt':
                    res_data_per_test_gpt[test_name] = res_data_gpt
        for i in idxs_source:
            if i not in idxs_gpt:
                res_data_per_test_source[test_name].pop(i)
    return res_data_per_test_source, res_data_per_test_gpt


def get_res_data(res_paths, dataset):
    '''

    :param res_paths: dict. keys:  'prompt_manipulation', 'image_manipulation'. values: path to res
    :return: res_data_per_test: dict. keys:  'prompt_manipulation', 'image_manipulation'. values: dict to res per image name and style
    '''
    res_data_per_test = {}
    i = 1 #0 - idx_img_name_in_res
    j = 2 # caption res in column idx=2
    for test_name in res_paths:
        if test_name.startswith('capdec'):
            i = 0
            j = 1
        res_data = {}
        # with open(res_paths[test_name], 'r') as csvfile:
        encoding_options = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encoding_options:
            try:
                with open(res_paths[test_name], 'r',encoding=encoding) as csvfile:
                    spamreader = csv.reader(csvfile)
                    title = True
                    styles = []
                    for row in spamreader:
                        if '.jpg' in row[i]:
                            k = row[i].split('.jpg')[0]
                        else:
                            k = row[i]
                        if 'COCO' in k:
                            k = k.split('_')[-1]
                        try:
                            if dataset == 'senticap':
                                k = int(k)
                        except:
                            pass
                        if title:
                            if row[1]=='factual':
                                i = 0
                                j = 1
                            try:
                                i = row.index('img_num')
                                j = row.index('img_num') + 1
                            except:
                                pass
                            styles = row[i+1:]
                            title = False
                            continue
                        else:
                            try:
                                res_data[k] = {}
                                for i_s, s in enumerate(styles):
                                    # res_data[k][s] = row[i+1]
                                    # limit sentence to target_seq_length as we create in ZeroStyleCap
                                    res_data[k][s] = ' '.join(row[i_s + j].split()[:target_seq_length])
                            except:
                                pass
                res_data_per_test[test_name] = res_data
                break  # Exit the loop if the file was read successfully
            except UnicodeDecodeError:
                continue  # Try the next encoding option
    return res_data_per_test


def write_results_for_all_frames(all_scores, tgt_eval_results_path_for_all_frames, metrics):
    print(f"Write results to {tgt_eval_results_path_for_all_frames}...")
    title = ['k', 'test_name', 'style']
    title.extend(metrics)
    title.extend(['res', 'gts', 'image_path'])
    with open(tgt_eval_results_path_for_all_frames, 'w') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(title)
        for k in all_scores:
            for test_name in all_scores[k]:
                for style in all_scores[k][test_name]:
                    row = [k, test_name, style]
                    for metric in metrics:
                        if metric in all_scores[k][test_name][style]:
                            row.append(all_scores[k][test_name][style][metric])
                        else:
                            row.append('')
                    if 'res' in all_scores[k][test_name][style]:
                        row.append(all_scores[k][test_name][style]['res'])
                    else:
                        row.append('')
                    if 'gts' in all_scores[k][test_name][style]:
                        row.append(all_scores[k][test_name][style]['gts'])
                    else:
                        row.append('')
                    if 'image_path' in all_scores[k][test_name][style]:
                        row.append(all_scores[k][test_name][style]['image_path'])
                    else:
                        row.append('')
                    writer.writerow(row)
    print(f"finished to write results for all frames in {tgt_eval_results_path_for_all_frames}")


def write_results(mean_score, tgt_eval_results_path, dataset, metrics, styles, vocab_size,  std_score = None, median_score=None):
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
        avg_metric_std = {}
        avg_metric_median = {}
        metrics.append("diversity-vocab_size")
        for test_name in mean_score: #go over each experiement
            row = [test_name] #first column
            for style in styles:
                for metric in metrics:
                    if test_name.startswith('capdec') and style not in vocab_size[test_name]:
                        continue
                    if metric=="diversity-vocab_size" or not np.isnan(mean_score[test_name][metric][style]):
                        if title:
                            dataset_row.append(dataset+'-avg')
                            style_row.append(style+'-avg')
                            metrics_row.append(metric+'-avg')
                            if metric != "diversity-vocab_size":
                                dataset_row.append(dataset)
                                dataset_row.append(dataset)
                                style_row.append(style)
                                style_row.append(style)
                                metrics_row.append(metric + '-std')
                                metrics_row.append(metric + '-median')
                        if metric == "diversity-vocab_size":
                            row.append(vocab_size[test_name][style])
                        else: # append score values
                            row.append(mean_score[test_name][metric][style])
                            row.append(std_score[test_name][metric][style])
                            row.append(median_score[test_name][metric][style])
                        if metric not in avg_metric:
                            if metric == "diversity-vocab_size":
                                val_for_metric = vocab_size[test_name][style]
                            else:
                                val_for_metric = mean_score[test_name][metric][style]
                                val_for_metric_std = std_score[test_name][metric][style]
                                val_for_metric_median = median_score[test_name][metric][style]
                                avg_metric_std[metric] = [val_for_metric_std]  # list for total avg over styles
                                avg_metric_median[metric] = [val_for_metric_median]  # list for total avg over styles
                            avg_metric[metric] = [val_for_metric] #list for total avg over styles
                        else:
                            if metric == "diversity-vocab_size":
                                val_for_metric = vocab_size[test_name][style]
                            else:
                                val_for_metric = mean_score[test_name][metric][style]
                                avg_metric_std[metric].append(std_score[test_name][metric][style])
                                avg_metric_median[metric].append(median_score[test_name][metric][style])
                            avg_metric[metric].append(val_for_metric)
                # if 'diversity-vocab_size' not in avg_metric:
                #     avg_metric['diversity-vocab_size'] = [vocab_size[test_name][style]]
                # else:
                #     if style in vocab_size[test_name]:
                #         avg_metric['diversity-vocab_size'].append(vocab_size[test_name][style])
            for m in avg_metric:
                if m != "diversity-vocab_size":
                    if title:
                        dataset_row.extend(['total avg.']*3)
                        style_row.extend(['total avg.']*3)
                        metrics_row.append(m)
                        metrics_row.append(m+'-std')
                        metrics_row.append(m+'-median')
                    row.append(np.mean(avg_metric[m]))
                    row.append(np.mean(avg_metric_std[m]))
                    row.append(np.mean(avg_metric_median[m]))
                else:
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
#         for test_name in all_scores[k]:
#             if test_name not in fluency_statistic:
#                 fluency_statistic[test_name] = {}
#             for style in all_scores[k][test_name]:
#                 if style not in fluency_statistic[test_name]:
#                     fluency_statistic[test_name][style] = [all_scores[k][test_name][style]['fluency']]
#                 else:
#                     fluency_statistic[test_name][style].append(all_scores[k][test_name][style]['fluency'])
#
#     print('finish to calc statistic of fluency')
#     # Generate data on commute times.
#     size, scale = 1000, 10
#     commutes = pd.Series(np.random.gamma(scale, size=size) ** 1.5)
#     commutes = pd.Series([1,1,2,2,3,3,3,3,3])
#     fig1 = plt.gcf()
#     for i, test_name in enumerate(fluency_statistic):
#         for j, style in enumerate(fluency_statistic[test_name]):
#             plt.subplot(len(fluency_statistic),len(fluency_statistic[test_name]),i*len(fluency_statistic[test_name])+j+1)
#
#             commutes = pd.Series(fluency_statistic[test_name][style])
#             commutes.plot.hist(grid=True, bins=20, rwidth=0.9,
#                                color='#607c8e')
#             plt.title(f'Fluency Histogram for: {test_name} - {style}')
#             plt.xlabel('Counts')
#             plt.ylabel('Commute Time')
#             plt.grid(axis='y', alpha=0.75)
#     plt.show(block=False)
#     os.makedirs(config['dir_path_for_eval_only_fluency'], exist_ok=True)
#     fig1.savefig(os.path.join(config['dir_path_for_eval_only_fluency'], f'_{test_name}_{style}.png'))
#     ##############

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default=os.path.join('.', 'configs', 'evaluation_config.yaml'),
                        help='full path to config file')
    parser.add_argument("--cuda_idx", type=int, default=0)
    args = parser.parse_args()
    return args

def get_final_results():
    #24/4/23 results from 23.3.23:
    #senticap:
    log_prompt_manipulation = "senticap_prompt_manipulation_debug.txt"
    prompt_manipulation = "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_embed_debug_loss/25_03_2023/d4jnwh7t-copper-cherry-13/results_23_36_32__25_03_2023.csv"

    log_image_manipulation = "senticap_image_manipulation_debug.txt"
    image_manipulation = "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_embed_debug_loss/25_03_2023/h9yfdeh2-misunderstood-forest-12/results_23_33_40__25_03_2023.csv"

    log_image_an_prompt_manipulation = "senticap_image_and_prompt_manipulation_debug.txt"
    image_an_prompt_manipulation = "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_embed_debug_loss/25_03_2023/jk7xzo7u-dulcet-bush-11/results_23_29_58__25_03_2023.csv"

    log_zerostylecap = "debug_loss_real_senticap_23_2_v0.txt"
    zerostylecap = "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_embed_debug_loss/23_03_2023/xuk6u1fz-eternal-night-7/results_21_29_00__23_03_2023.csv"

    # flickrstyle10k:
    log_prompt_manipulation = ["flickr_prompt_manipulation_debug_25_3_23.txt",\
                               "flickr_prompt_manipulation_debug_29_3_23_v0.txt",\
                           "flickr_prompt_manipulation_debug_29_3_23_v1.txt",\
                           "flickr_prompt_manipulation_debug_29_3_23_v2.txt",\
                           "flickr_prompt_manipulation_debug_29_3_23_v3.txt",\
                           "flickr_prompt_manipulation_debug_25_3_23_v2.txt"]
    prompt_manipulation = ["/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/25_03_2023/n1knhbfh-dainty-cosmos-50/results_23_46_15__25_03_2023.csv",\
                           "home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/29_03_2023/j8pq7aru-bright-bee-60/results_12_49_19__29_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/29_03_2023/1m1aqhch-azure-leaf-59/results_12_49_19__29_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/29_03_2023/mt0pvyea-upbeat-vortex-61/results_12_49_29__29_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/29_03_2023/chp8yi87-ancient-fire-62/results_12_49_53__29_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/25_03_2023/19i488ig-ancient-brook-51/results_23_47_58__25_03_2023.csv"]

    log_image_manipulation = ["flickr_image_manipulation_debug_25_3_23_v0.txt",\
                           "flickr_image_manipulation_debug_25_3_23_v1.txt",\
                           "flickr_image_manipulation_debug_25_3_23_v2.txt"]
    image_manipulation = ["/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/25_03_2023/k350v2ad-sandy-grass-56/results_23_58_20__25_03_2023.csv",\
                           "home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/25_03_2023/pbdt09l9-fast-eon-57/results_23_59_05__25_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/26_03_2023/bc466z75-sage-grass-58/results_00_00_40__26_03_2023.csv"]

    log_image_an_prompt_manipulation = ["flickr_image_and_prompt_manipulation_debug_25_3_23_v0.txt",\
                           "flickr_image_and_prompt_manipulation_debug_25_3_23_v1.txt",\
                           "flickr_image_and_prompt_manipulation_debug_25_3_23_v2.txt"]
    image_an_prompt_manipulation = ["/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/25_03_2023/l00kj482-rural-cloud-52/results_23_50_52__25_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/25_03_2023/3b5cwh3h-comic-bush-54/results_23_54_41__25_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/25_03_2023/9olhnst5-electric-terrain-55/results_23_55_39__25_03_2023.csv"]

    log_zerostylecap = ["debug_loss_flickr_23_2_v0.txt",\
                           "debug_loss_flickr_23_2_v1.txt",\
                           "debug_loss_flickr_23_2_v3.txt",\
                           "debug_loss_flickr_23_2_v4.txt",\
                           "debug_loss_flickr_23_2_v5.txt",\
                           "debug_loss_flickr_23_2_v6.txt",\
                           "debug_loss_flickr_23_2_v7.txt"]
    zerostylecap = ["/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/23_03_2023/az2h2lu8-lucky-night-34/results_13_37_13__23_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/23_03_2023/l8ep47kj-graceful-water-33/results_13_32_30__23_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/23_03_2023/zm963t5c-iconic-pyramid-47/results_21_40_14__23_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/23_03_2023/rakysj3j-firm-water-45/results_21_35_24__23_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/23_03_2023/i3x63emn-unique-universe-49/results_21_52_42__23_03_2023.csv",\
                           "home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/23_03_2023/kc5j6ra3-young-shape-43/results_13_44_10__23_03_2023.csv",\
                           "/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/23_03_2023/nucvqzc2-divine-serenity-44/results_13_44_14__23_03_2023.csv"]



def main():
    get_final_results()
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
    gts_per_data_set = get_gts_data(config['annotations_path'], config['imgs_path'], config['data_split'],
                                    factual_captions, config['max_num_imgs2test'])

    res_data_per_test = get_res_data(config['res_path2eval'],config['dataset'])
    if True:
    # #todo: remove
    # print("!!!!!!!!!!!!!!remove!!!!!!!!!!!!!!!")
    # res_data_per_test_source, res_data_per_test_gpt = get_res_data_GPT(config['res_path2eval'])
    # for res_data_idx, res_data_per_test in enumerate([res_data_per_test_source, res_data_per_test_gpt]):
    #     if res_data_idx == 0:
    #         prefix_file_name = 'source_'
    #     elif res_data_idx == 1:
    #         prefix_file_name = 'gpt_'
        # copy_imgs_to_test_dir(gts_per_data_set, res_data_per_test, styles, metrics, gt_imgs_for_test)
        # exit(0)
        mean_score, all_scores, std_score, median_score = calc_score(gts_per_data_set, res_data_per_test, config['styles'], config['metrics'],
                                            config['cuda_idx'], data_dir, config['txt_cls_model_paths'],
                                            config['labels_dict_idxs'], gt_imgs_for_test, config)

        vocab_size = diversitiy(res_data_per_test, gts_per_data_set)
        # analyze_fluency(all_scores,config)

        for test_name in res_data_per_test:
            for metric in config['metrics']:
                for style in config['styles']:
                    print(f"{test_name}: {config['dataset']}: {metric} mean score for {style} = {mean_score[test_name][metric][style]}")
                    print(f"{test_name}: {config['dataset']}: {metric} std score for {style} = {std_score[test_name][metric][style]}")
                    print(f"{test_name}: {config['dataset']}: {metric} median score for {style} = {median_score[test_name][metric][style]}")
        for test_name in res_data_per_test:
            print(f'Vocabulary size for experiment {test_name} dataset is {vocab_size[test_name]}')

        test_name = list(config['res_path2eval'].values())[0].rsplit('/', 1)[1]
        # tgt_eval_results_file_name = os.path.join(list(config['res_path2eval'].values())[0].rsplit('/', 1)[0],
        #                                           config['tgt_eval_results_file_name'])
        # tgt_eval_results_file_name_for_all_frames = os.path.join(
        #     list(config['res_path2eval'].values())[0].rsplit('/', 1)[0],
        #     config['tgt_eval_results_file_name_for_all_frames'])
        #
        tgt_eval_results_file_name = os.path.join(list(config['res_path2eval'].values())[0].rsplit('/', 1)[0],
                                                  config['tgt_eval_results_file_name'].split('.')[0]+'_'+test_name)
        tgt_eval_results_file_name_for_all_frames = os.path.join(
            list(config['res_path2eval'].values())[0].rsplit('/', 1)[0],
            config['tgt_eval_results_file_name_for_all_frames'].split('.')[0]+'_'+test_name)

    # todo: remove
        # print("!!!!!!!!!!!!!!remove!!!!!!!!!!!!!!!")
        # tgt_eval_results_file_name = os.path.join(list(config['res_path2eval'].values())[0].rsplit('/', 1)[0],
        #                                           prefix_file_name+config['tgt_eval_results_file_name'])
        #
        # tgt_eval_results_file_name_for_all_frames = os.path.join(
        #     list(config['res_path2eval'].values())[0].rsplit('/', 1)[0],
        #     prefix_file_name+config['tgt_eval_results_file_name_for_all_frames'])



        print(f"finished to evaluat on {len(all_scores)} images.")
        write_results(mean_score, tgt_eval_results_file_name, config['dataset'], config['metrics'].copy(), config['styles'],
                      vocab_size, std_score, median_score)
        write_results_for_all_frames(all_scores, tgt_eval_results_file_name_for_all_frames, config['metrics'])

    print('Finished to evaluate')


if __name__ == '__main__':
    main()
