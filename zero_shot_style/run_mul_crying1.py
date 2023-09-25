import json
import timeit
import os
# import pdb

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print(f"cuda={torch.cuda.is_available()}")
import pandas as pd
# import torch
import clip
import wandb
import yaml
from zero_shot_style.evaluation.evaluation_all import get_gts_data, CLIPScoreRef, CLIPScore, STYLE_CLS, STYLE_CLS_EMOJI, \
    STYLE_CLS_ROBERTA, CLAPScore
from zero_shot_style.evaluation.evaluation_all import evaluate_single_res
from zero_shot_style.evaluation.pycocoevalcap.bleu.bleu import Bleu
from zero_shot_style.evaluation.pycocoevalcap.rouge.rouge import Rouge

from zero_shot_style.writer import write_results, write_results_of_text_style, write_results_image_manipulation, \
    write_evaluation_results, write_results_of_text_style_all_models, write_caption_results, write_debug_tracking, \
    write_img_idx_to_name

from model.ZeroCLIP import CLIPTextGenerator
from datetime import datetime
import os.path
import csv
from collections import defaultdict
import numpy as np
import pickle
from datetime import datetime
from utils import parser, get_hparams
from evaluate import load

MAX_PERPLEXITY = 1500
DEFAULT_PERPLEXITY_SCORE = 1
DEFAULT_CLIP_SCORE = 1
DEFAULT_CLAP_SCORE = 1
DEFAULT_STYLE_CLS_SCORE = 1
DEFAULT_STYLE_CLS_EMOJI_SCORE = 1
EPSILON = 0.0000000001
MAX_NUM_IMGS_TO_TEST = 1000

def get_args():
    parser.add_argument('--config_file', type=str,
                        # default=os.path.join('.', 'configs', 'config.yaml'),
                        # default=os.path.join('.', 'configs', 'config_mul_clip_style_v1_romantic.yaml'), #todo: change config file
                        # default=os.path.join('.', 'configs', 'config_3_loss_roberta_v101neg_test.yaml'), #todo: change config file
                        # default=os.path.join('.', 'configs', 'config_mul_clip_style_roberta_v20neg_test.yaml'), #todo: change config file
                        # default=os.path.join('.', 'configs', 'config_update_vit15pos_test_best_fluence.yaml'), #todo: change config file
                        # default=os.path.join('.', 'configs', 'config_update_vit17neg_check.yaml'), #todo: change config file
                        # default=os.path.join('.', 'configs', 'config_audio.yaml'), #todo: change config file
                        default=os.path.join('.', 'configs', 'config_audio_crying1.yaml'), #todo: change config file
                        # default=os.path.join('.', 'configs', 'config_update_vit_audio.yaml'), #todo: change config file
                        help='full path to config file')
    # parser = argparse.ArgumentParser() #comment when using, in addition, the arguments from zero_shot_style.utils
    # parser.add_argument('--wandb_mode', type=str, default='disabled', help='disabled, offline, online')
    parser.add_argument("--img_name", type=int, default=0)
    parser.add_argument("--use_all_imgs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo or gpt-j")
    parser.add_argument("--clip_checkpoints", type=str, default="~/projects/zero-shot-style/zero_shot_style/clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    parser.add_argument("--data_type", type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument("--cond_text", type=str, default="Image of a")
    # parser.add_argument("--cond_text_list", nargs="+", type=str, default=["Image of a"])
    # parser.add_argument("--cond_text_list", nargs="+", type=str, default=["A creative short caption I can generate to describe this image is:",
    #                                                                       "A creative positive short caption I can generate to describe this image is:",
    #                                                                       "A creative negative short caption I can generate to describe this image is:",
    #                                                                       "A creative humoristic short caption I can generate to describe this image is:",
    #                                                                       "A creative romantic short caption I can generate to describe this image is:"])
    # parser.add_argument("--cond_text", type=str, default="")
    parser.add_argument("--cond_text2", type=str, default="")
    parser.add_argument("--reset_context_delta", action="store_true",
                        help="Should we reset the context at each token gen")
    parser.add_argument("--reverse_imgs_list", action="store_true",
                        help="Should we reverse the order of images list we run on")
    parser.add_argument("--update_ViT", action="store_true",
                        help="Should update CLIP tensors also")

    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--audio_temperature", type=float, default=1)
    parser.add_argument("--std_embedding_vectors_positive", type=float, default=0.028914157)
    parser.add_argument("--std_embedding_vectors_negative", type=float, default=0.020412436)
    parser.add_argument("--desired_min_CLIP_score", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--text_style_scale", type=float, default=1)
    parser.add_argument("--sentiment_temperature", type=float, default=0.01)
    parser.add_argument("--threshold_sentiment", type=float, default=0.3597)
    parser.add_argument("--requires_min_style_score", type=float, default=0.35 )
    parser.add_argument("--requires_min_clip_score_val", type=float, default=0.26)
    parser.add_argument("--requires_num_min_clip_score_val", type=int, default=10)
    parser.add_argument("--stepsize_clip", type=float, default=0.3)

    parser.add_argument("--requires_min_clip_score_val_pos", type=float, default=0.35)
    parser.add_argument("--requires_min_clip_score_val_neg", type=float, default=0.35)
    parser.add_argument("--requires_min_style_score_pos", type=float, default=0.26)
    parser.add_argument("--requires_min_style_score_neg", type=float, default=0.26)
    parser.add_argument("--requires_num_min_clip_score_val_pos", type=int, default=10)
    parser.add_argument("--requires_num_min_clip_score_val_neg", type=int, default=10)

    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=2)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--max_num_of_imgs", type=int, default=2)  # 1e6)
    parser.add_argument("--evaluation_metrics", nargs="+",
                        default=['CLIPScore', 'fluency', 'style_classification'])
                        # default=['CLIPScore', 'fluency', 'style_classification', 'style_classification_emoji'])
    # default=['bleu', 'rouge', 'clip_score_ref', 'CLIPScore', 'fluency', 'style_classification'])

    parser.add_argument("--cuda_idx_num", type=str, default="0")
    parser.add_argument("--img_idx_to_start_from", type=int, default=0)

    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'arithmetics', 'img_prompt_manipulation'])

    # parser.add_argument("--dataset", type=str, default="senticap")  # todo: add: "flickrstyle10k"])

    parser.add_argument("--imgs_dict", type=str, default=os.path.join(os.path.expanduser('~'), 'data', 'senticap'),
                        help="Path to images dict for captioning")

    parser.add_argument("--caption_img_path", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'data', 'imgs', '101.jpeg'),
                        help="Path to image for captioning")

    parser.add_argument("--arithmetics_imgs", nargs="+",
                        default=['example_images/arithmetics/woman2.jpg',
                                 'example_images/arithmetics/king2.jpg',
                                 'example_images/arithmetics/man2.jpg'])

    parser.add_argument("--arithmetics_style_imgs", nargs="+",
                        default=['49', '50', '51', '52', '53'])
    parser.add_argument("--specific_img_idxs_to_test", nargs="+",
                        default=[])
    parser.add_argument("--specific_imgs_to_test", nargs="+",
                        default=[])

    parser.add_argument("--arithmetics_weights", nargs="+", default=[1, 1, -1])
    parser.add_argument("--use_style_model", action="store_true", default=False)
    parser.add_argument("--use_audio_model", action="store_true", default=False)
    parser.add_argument("--use_img_path", type=str, help="path to specific image")
    parser.add_argument("--use_text_style_example", action="store_true", default=False)

    parser.add_argument("--desired_improvement_loss", type=float, default=0.01)
    parser.add_argument("--th_clip_loss", type=float, default=35)
    parser.add_argument("--th_ce_loss", type=float, default=12)
    parser.add_argument("--th_style_loss", type=float, default=35.5)

    parser.add_argument("--num_iterations_clip_style",  type=int, default=2)
    parser.add_argument("--loss_scale_style_clip", type=float, default=0.5)
    parser.add_argument("--loss_scale_src_clip_clip", type=float, default=0.5)

    args = parser.parse_args()

    return args


def run(config, img_path, desired_style_embedding_vector, desired_style_embedding_vector_std, cuda_idx, title2print,
        model_path, style_type, tmp_text_loss, label, img_dict, debug_tracking, text_generator=None,
        image_features=None, evaluation_obj=None, desired_style_bin = False, clip_img=None):
    # debug_tracking: debug_tracking[img_path][label][word_num][iteration][module]:<list>
    if text_generator == None:
        text_generator = CLIPTextGenerator(cuda_idx=cuda_idx, model_path=model_path, tmp_text_loss=tmp_text_loss,
                                           text_style_scale=config['text_style_scale'], config=config, evaluation_obj=evaluation_obj,img_path = img_path,**vars(config))
    if image_features == None:
        image_features,clip_img = text_generator.get_img_feature([img_path], None, return_k_v=False, get_preroccessed_img=True,kv_only_first_layer=config.get('kv_only_first_layer',True))

    # SENTIMENT: added scale parameter
    if config['imitate_text_style'] or config['use_text_style_example']:
        text_style = label
    else:
        text_style = None
    t1 = timeit.default_timer();

    captions = text_generator.run(image_features, config['cond_text'], config['beam_size'], config['text_style_scale'],
                                  text_style, desired_style_embedding_vector, desired_style_embedding_vector_std,
                                  style_type, img_idx=config['img_path_idx'], img_name=img_path.split('/')[-1], style=label, desired_style_bin=config['labels_dict_idxs'][label], clip_img=clip_img)
    debug_tracking[img_path][label] = text_generator.get_debug_tracking()
    t2 = timeit.default_timer();

    if config['model_based_on'] == 'bert':
        encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in
                            captions]
    elif config['model_based_on'] == 'clip':  # for text_style
        encoded_captions = [text_generator.text_style_model(c) for c in captions]

    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()
    device = f"cuda" if torch.cuda.is_available() else "cpu"  # todo: change
    clip_grades = (torch.cat(encoded_captions) @ image_features.t()).squeeze()

    if evaluation_obj and ('CLAPScore' in evaluation_obj):
        style_cls_grades = torch.tensor(
            evaluation_obj['CLAPScore'].compute_score_for_list(captions))
        style_cls_grades = style_cls_grades.to(device)
        best_harmonic_mean_idx = (
                    len(captions) * clip_grades * style_cls_grades / (clip_grades + style_cls_grades)).argmax()

    if evaluation_obj and ('style_classification' in evaluation_obj or 'style_classification_roberta' in evaluation_obj):
        if 'style_classification' in evaluation_obj:
            style_cls_grades = torch.tensor(evaluation_obj['style_classification'].compute_label_for_list(captions,label)).to(device)
        elif 'style_classification_roberta' in evaluation_obj:
            style_cls_grades = torch.tensor(
                evaluation_obj['style_classification_roberta'].compute_label_for_list(captions, label)).to(device)
        #check if there is caption with correct style
        consider_style = False
        for i in style_cls_grades:
            if type(i)==type((1,1)):
                i = i[0]
            if i>0:
                consider_style = True
                break
        if consider_style:
            style_cls_grades = torch.tensor(style_cls_grades)
            # calc harmonic average:
            best_harmonic_mean_idx = (len(captions) * clip_grades * style_cls_grades / (clip_grades + style_cls_grades)).argmax()
        else:
            best_harmonic_mean_idx = best_clip_idx
    # if config['style_type'] == 'emoji':
    #     device = f"cuda" if torch.cuda.is_available() else "cpu"  # todo: change
    #     tokenized, _, _ = text_generator.emoji_st_tokenizer.tokenize_sentences(captions)
    #     tokenized = torch.from_numpy(tokenized.astype(np.int32))
    #     emoji_style_probs = torch.tensor(text_generator.emoji_style_model(tokenized)).to(device)
    #     emoji_style_grades = emoji_style_probs[:, text_generator.config['idx_emoji_style_dict'][text_generator.style]].sum(-1)
    #     #calc harmonic average:
    #     best_harmonic_mean_idx = (len(captions)*clip_grades*emoji_style_grades/(clip_grades+emoji_style_grades)).argmax()
    #     # emoji_style_grades_normalized = emoji_style_grades / torch.sum(emoji_style_grades)
    else:
        best_harmonic_mean_idx = best_clip_idx
    print(captions)

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    new_title2print = f'~~~~~~~~\n{dt_string} | Work on img path:' + title2print.split(' | Work on img path:')[1]
    print(new_title2print)

    print('best clip:', config['cond_text'] + captions[best_harmonic_mean_idx])
    print(f"Time to create caption is: {(t2 - t1) / 60} minutes = {t2 - t1} seconds.")
    img_dict[img_path][style_type][config['text_style_scale']][label] = config['cond_text'] + captions[best_harmonic_mean_idx]
    return config['cond_text'] + captions[best_harmonic_mean_idx]


def run_arithmetic(text_generator, config, model_path, img_dict_img_arithmetic, base_img, dataset_type, imgs_path,
                   img_weights, cuda_idx, title2print,img_name=None, style=None):
    if text_generator == None:
        text_generator = CLIPTextGenerator(cuda_idx=cuda_idx, model_path=model_path, config=config, img_idx=config['img_path_idx'], **vars(config))
    # text_generator = CLIPTextGenerator(cuda_idx=cuda_idx, **vars(config))

    image_features = text_generator.get_combined_feature(imgs_path, [], img_weights, None)
    t1 = timeit.default_timer();
    captions = text_generator.run(image_features, config['cond_text'], beam_size=config['beam_size'],img_idx=config['img_path_idx'], img_name=img_name, style=style)

    t2 = timeit.default_timer();

    if config['model_based_on'] == 'bert':
        encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in
                            captions]
    elif config['model_based_on'] == 'clip':  # for text_style
        encoded_captions = [text_generator.text_style_model(c) for c in captions]

    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    new_title2print = f'~~~~~~~~\n{dt_string} | Work on img path:' + title2print.split(' | Work on img path:')[1]
    print(new_title2print)

    print('best clip:', config['cond_text'] + captions[best_clip_idx])
    print(f"Time to create caption is: {(t2 - t1) / 60} minutes = {t2 - t1} seconds.")
    img_dict_img_arithmetic[base_img][dataset_type] = config['cond_text'] + captions[best_clip_idx]

    return config['cond_text'] + captions[best_clip_idx]


def run_img_and_prompt_manipulation(config, img_dict_img_arithmetic, base_img, dataset_type, imgs_path, img_weights,
                                    cuda_idx, title2print):
    # text_generator = CLIPTextGenerator(**vars(args))
    text_generator = CLIPTextGenerator(cuda_idx=cuda_idx, config=config, **vars(config))

    image_features = text_generator.get_combined_feature(imgs_path, [], img_weights, None)
    t1 = timeit.default_timer()
    captions = text_generator.run(image_features, config['cond_text'], beam_size=config['beam_size'])
    t2 = timeit.default_timer()

    if config['model_based_on'] == 'bert':
        encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in
                            captions]
    elif config['model_based_on'] == 'clip':  # for text_style
        encoded_captions = [text_generator.text_style_model(c) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    new_title2print = f'~~~~~~~~\n{dt_string} | Work on img path:' + title2print.split(' | Work on img path:')[1]
    print(new_title2print)

    print('best clip:', config['cond_text'] + captions[best_clip_idx])
    print(f"Time to create caption is: {(t2 - t1) / 60} minutes = {t2 - t1} seconds.")
    img_dict_img_arithmetic[base_img][dataset_type] = config['cond_text'] + captions[best_clip_idx]

    return config['cond_text'] + captions[best_clip_idx]


def get_title2print(caption_img_path, dataset_type, label, config):
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    title2print = f"~~~~~~~~\n{dt_string} | Work on img path: {caption_img_path} with:" \
                  f"\nresults dir= *** {config['experiment_dir']} ***" \
                  f"\ndataset_type= *** {dataset_type} ***" \
                  f"\nstyle of: *** {label} ***" \
                  f"\n~~~~~~~~"
    return title2print


def get_full_path_of_stylized_images(data_dir, i):
    if os.path.isfile(os.path.join(data_dir, 'stylized_images',
                                   str(i) + ".jpeg")):
        return os.path.join(data_dir, 'stylized_images',
                            str(i) + ".jpeg")
    elif os.path.isfile(os.path.join(data_dir, 'stylized_images',
                                     str(i) + ".jpg")):
        return os.path.join(data_dir, 'stylized_images',
                            str(i) + ".jpg")
    elif os.path.isfile(os.path.join(data_dir, 'stylized_images',
                                     str(i) + ".png")):
        return os.path.join(data_dir, 'stylized_images',
                            str(i) + ".png")
    else:
        return None


def calculate_avg_score_wo_fluence(clip_score, style_cls_score):
    avg_total_score = 2 * (style_cls_score * clip_score) / (
                style_cls_score + clip_score)
    return avg_total_score


def calculate_avg_score(clip_score, fluency_score, style_cls_score):
    avg_total_score = 3 * (style_cls_score * clip_score * fluency_score) / (
                style_cls_score + clip_score + fluency_score)
    return avg_total_score

def calculate_avg_score_clip_gpt(clip_score, fluency_score):
    avg_total_score = 2 * (clip_score * fluency_score) / ( clip_score + fluency_score)
    return avg_total_score


def get_img_full_path(base_path, i):
    if os.path.isfile(os.path.join(base_path, 'data', 'imgs',
                                   str(i) + ".jpeg")):
        return os.path.join(base_path, 'data', 'imgs',
                            str(i) + ".jpeg")
    elif os.path.isfile(os.path.join(base_path, 'data', 'imgs',
                                     str(i) + ".jpg")):
        return os.path.join(base_path, 'data', 'imgs',
                            str(i) + ".jpg")
    elif os.path.isfile(os.path.join(base_path, 'data', 'imgs',
                                     str(i) + ".png")):
        return os.path.join(base_path, 'data', 'imgs',
                            str(i) + ".png")
    else:
        return None


class Caption:
    def __init__(self, img_name, style, res_caption_text, gt_caption_text, img_path, classification, clip_score,
                 fluency, avg_total_score, factual_captions, style_cls_score, style_cls_score_emoji,clap_score=1):
        self.img_name = img_name
        self.style = style
        self.res_caption_text = res_caption_text
        self.gt_caption_text = gt_caption_text
        self.img_path = img_path
        self.perplexity = fluency
        self.avg_style_cls_score = style_cls_score
        self.avg_style_cls_score_emoji = style_cls_score_emoji
        self.avg_clip_score = clip_score
        self.avg_clap_score = clap_score
        self.avg_fluency_score = fluency
        self.classification = classification
        self.avg_total_score = avg_total_score
        self.factual_captions = factual_captions

    def get_img_name(self):
        return self.img_name

    def get_style(self):
        return self.style

    def get_caption_text(self):
        return self.res_caption_text

    def get_gt_caption_text(self):
        return self.gt_caption_text

    def get_factual_captions(self):
        return self.factual_captions

    def get_img_path(self):
        return self.img_path

    def get_classification(self):
        return self.classification

    def get_style_cls_score(self):
        return self.avg_style_cls_score

    def get_style_cls_emoji_score(self):
        return self.avg_style_cls_score_emoji

    def get_clip_score(self):
        return self.avg_clip_score

    def get_clap_score(self):
        return self.avg_clap_score

    def get_fluency_score(self):
        return self.avg_fluency_score

    def get_total_score(self):
        return self.avg_total_score

    def set_classification(self, classification):
        self.classification = classification

    def set_style_cls_score(self, avg_style_cls_score):
        self.avg_style_cls_score = avg_style_cls_score

    def set_style_cls_emoji_score(self, avg_style_cls_emoji_score):
        self.avg_style_cls_emoji_score = avg_style_cls_emoji_score

    def set_clip_score(self, avg_clip_score):
        self.avg_clip_score = avg_clip_score

    def set_clap_score(self, avg_clap_score):
        self.avg_clap_score = avg_clap_score

    def set_fluency_score(self, avg_fluency_score):
        self.avg_fluency_score = avg_fluency_score

    def set_total_score(self, avg_total_score):
        self.avg_total_score = avg_total_score

class Fluency:
    def __init__(self, desired_labels):
        self.model_id = 'gpt2'
        self.perplexity = load("perplexity", module_type="measurement")
        self.tests = []
        self.img_names = []
        self.styles = []
        self.labels = desired_labels

    def add_test(self, res, img_name, style):
        self.tests.append(res)
        self.img_names.append(img_name)
        self.styles.append(style)

    def compute_score(self):
        results = self.perplexity.compute(data=self.tests, model_id=self.model_id, add_start_token=False)
        perplexities = {}
        total_perplexities = []
        for i, p in enumerate(results['perplexities']):
            if self.img_names[i] not in perplexities:
                perplexities[self.img_names[i]] = {}
            fixed_perplexity = 1 - np.min([p, MAX_PERPLEXITY]) / MAX_PERPLEXITY
            perplexities[self.img_names[i]][self.styles[i]] = fixed_perplexity
            total_perplexities.append(fixed_perplexity)
        total_avg_perplexity = np.mean(total_perplexities)
        # total_avg_perplexity = 1-np.min([results['mean_perplexity'],MAX_PERPLEXITY])/MAX_PERPLEXITY
        return perplexities, total_avg_perplexity


    def compute_score_for_single_text(self, text):
        results = self.perplexity.compute(data=[text], model_id=self.model_id, add_start_token=False)
        fixed_perplexity = 1 - np.min([results['mean_perplexity'], MAX_PERPLEXITY]) / MAX_PERPLEXITY
        return results['perplexities'], fixed_perplexity

    def add_results(self, evaluation_results):
        for img_name in evaluation_results:
            for label in evaluation_results[img_name]:
                if label in self.labels:
                    self.add_test(evaluation_results[img_name][label]['res'], img_name, label)

def get_table_for_wandb(data_list):
    data = [[x, y] for (x, y) in zip(data_list, list(range(len(data_list))))]
    table = wandb.Table(data=data, columns=["x", "y"])
    return table


def get_factual_captions(factual_file_path_list):
    '''
    get dict of all factual captions
    :param factual_file_path_list: list of file path to factual data
    :return: factual_captions:dict:key=img_name,value=list of factual captions
    '''
    print("create dict of factual captions...")
    factual_captions = {}
    finish_im_ids = []
    for factual_file_path in factual_file_path_list:
        data = json.load(open(factual_file_path, "r"))
        for d in data['annotations']:
            if d['image_id'] in finish_im_ids:
                continue
            if d['image_id'] not in factual_captions:
                factual_captions[d['image_id']] = [d['caption']]
            else:
                factual_captions[d['image_id']].append(d['caption'])
        finish_im_ids = list(factual_captions.keys())
    return factual_captions


def get_list_of_imgs_for_caption(config):
    print("take list of images for captioning...")
    imgs_to_test = []
    print(f"config['max_num_of_imgs']: {config['max_num_of_imgs']}")
    if 'specific_img_idxs_to_test' in config and len(config['specific_img_idxs_to_test']) > 0:
        imgs_list = os.listdir(
            os.path.join(os.path.join(os.path.expanduser('~'), 'data', config['dataset']), 'images',
                         config['data_type']))
        for i in config['specific_img_idxs_to_test']:
            if i < len(imgs_list):
                i = int(i)
                im = imgs_list[i]
                imgs_to_test.append(
                    os.path.join(os.path.join(os.path.expanduser('~'), 'data', config['dataset']), 'images',
                                 config['data_type'], im))
        return imgs_to_test
    for i, im in enumerate(os.listdir(
            os.path.join(os.path.join(os.path.expanduser('~'), 'data', config['dataset']), 'images',
                         config['data_type']))):
        if len(imgs_to_test) >= int(config['max_num_of_imgs']) > 0:
            break
        if 'specific_idxs_to_skip' in config and len(config['specific_idxs_to_skip']) > 0 and i in config['specific_idxs_to_skip']:
            continue
        if ('.jpg' or '.jpeg' or '.png') not in im:
            continue
        if config.get('dataset','senticap') == 'senticap':
            if 'specific_imgs_to_test' in config and len(config['specific_imgs_to_test']) > 0 and int(
                    im.split('.')[0]) not in config['specific_imgs_to_test']:
                continue
        elif config.get('dataset','senticap') == 'flickrstyle10k':
            if 'specific_imgs_to_test' in config and len(config['specific_imgs_to_test']) > 0 and im.split('.')[0] not in config['specific_imgs_to_test']:
                continue
        imgs_to_test.append(os.path.join(os.path.join(os.path.expanduser('~'), 'data', config['dataset']), 'images',
                                         config['data_type'], im))
    print(f"***There are {len(imgs_to_test)} images to test.***")
    return imgs_to_test


def update_running_params(label, config):
    """Assume that in the configuration of the running there is single value of use_style_model, otherwise will be
    mistake in running params """

    def save_running_params(config):
        if 'running_param_clip_scale' not in config:
            config['running_param_use_style_model'] = config['use_style_model']
            config['running_param_clip_scale'] = config['clip_scale']
            config['running_param_ce_scale'] = config['ce_scale']
            config['running_param_text_style_scale'] = config['text_style_scale']
            config['running_param_beam_size'] = config['beam_size']
            config['running_param_num_iterations'] = config['num_iterations']
        return config

    def update_zerocap_params(config):
        save_running_params(config)
        config['clip_scale'] = config['zerocap_clip_scale']
        config['ce_scale'] = config['zerocap_ce_scale']
        config['beam_size'] = config['zerocap_beam_size']
        config['num_iterations'] = config['zerocap_num_iterations']
        config['text_style_scale'] = config['zerocap_text_style_scale']
        return config

    def update_running_params(config):
        save_running_params(config)
        config['use_style_model'] = config['running_param_use_style_model']
        config['clip_scale'] = config['running_param_clip_scale']
        config['ce_scale'] = config['running_param_ce_scale']
        config['text_style_scale'] = config['running_param_text_style_scale']
        config['beam_size'] = config['running_param_beam_size']
        config['num_iterations'] = config['running_param_num_iterations']
        return config

    if label == 'factual':
        config = update_zerocap_params(config)  # todo:check if need to return it
        config['use_style_model'] = False
    else:  # label=positive,negative...
        if config['use_style_model'] or config['cut_cand2clip']:
            config = update_running_params(config)
        else:
            config = update_zerocap_params(config)
    return config


def get_evaluation_obj(config, text_generator, evaluation_obj):
    if not evaluation_obj:
        evaluation_obj = {}
    if 'evaluation_metrics' not in config:
        return evaluation_obj
    if config["calc_evaluation"]:
        for metric in config['evaluation_metrics']:
            if metric == 'bleu' and 'bleu' not in evaluation_obj:
                evaluation_obj['bleu'] = Bleu(n=4)
            if metric == 'rouge' and 'rouge' not in evaluation_obj:
                evaluation_obj['rouge'] = Rouge()
            if metric == 'clip_score_ref' and 'clip_score_ref' not in evaluation_obj:
                evaluation_obj['clip_score_ref'] = CLIPScoreRef(text_generator)
            if metric == 'CLIPScore' and 'CLIPScore' not in evaluation_obj:
                evaluation_obj['CLIPScore'] = CLIPScore(text_generator)
            if metric == 'CLAPScore' and 'CLAPScore' not in evaluation_obj:
                evaluation_obj['CLAPScore'] = CLAPScore(config['audio_model_sampling_rate'], config['audio_sampling_rate'],config['audio_path'])
            if metric == 'fluency' and 'fluency' not in evaluation_obj:
                evaluation_obj['fluency'] = Fluency(config['desired_labels'])

            if metric == 'style_classification' and 'style_classification' not in evaluation_obj:
                txt_cls_model_path = os.path.join(os.path.expanduser('~'), config['txt_cls_model_path'])
                evaluation_obj['style_classification'] = STYLE_CLS(txt_cls_model_path, config['cuda_idx_num'],
                                                                   config['labels_dict_idxs'], None, config[
                                                                       'hidden_state_to_take_txt_cls'],config['max_batch_size_style_cls'])
                print(f"style_cls_obj = STYLE_CLS")
            if metric == 'style_classification_roberta' and 'style_classification_roberta' not in evaluation_obj:
                evaluation_obj['style_classification_roberta'] = STYLE_CLS_ROBERTA(config['finetuned_roberta_config'],
                                                  config['finetuned_roberta_model_path'], config['cuda_idx_num'], config['labels_dict_idxs_roberta'],
                                                  None)
                print(f"style_cls_obj = STYLE_CLS_ROBERTA")


            if metric == 'style_classification_emoji' and 'style_classification_emoji' not in evaluation_obj:
                evaluation_obj['style_classification_emoji'] = STYLE_CLS_EMOJI(config['emoji_vocab_path'],
                                                                               config['maxlen_emoji_sentence'],
                                                                               config['emoji_pretrained_path'],
                                                                               config['idx_emoji_style_dict'], config['use_single_emoji_style'], config['desired_labels'])

    return evaluation_obj


def evaluate_results(config, evaluation_results, gts_data, results_dir, factual_captions, text_generator, evaluation_obj):
    if 'evaluation_metrics' not in config:
        return
    print("Calc evaluation of the results...")
    # calc perplexity
    # if 'fluency' in config['evaluation_metrics'] and config['calc_fluency']:
    #     evaluation_obj['fluency'] = Fluency(config['desired_labels'])
    #     evaluation_obj['fluency'].add_results(evaluation_results)
    #     perplexities, mean_perplexity = evaluation_obj['fluency'].compute_score()
    # else:
    #     mean_perplexity = DEFAULT_PERPLEXITY_SCORE

    # evaluation_obj = get_evaluation_obj(config, text_generator, evaluation_obj)

    style_cls_scores = []
    style_cls_emoji_scores = []
    clip_scores = []
    clap_scores = []
    fluency_scores = []
    total_captions = []
    total_res_text = []
    total_gt_text = []
    avg_total_scores = []
    for img_name in evaluation_results:
        for label in list(evaluation_results[img_name].keys()):
            if label == 'img_path':
                continue
            # if config["dataset"] == "senticap":
            # evaluation_results[img_name][label]['gt'] = gts_data[img_name][label]  # todo: handle style type
            # evaluation_results[img_name][label]['scores'] = evaluate_single_res(
            #     evaluation_results[img_name][label]['res'], evaluation_results[img_name][label]['gt'],
            #     evaluation_results[img_name]['img_path'], label, config['evaluation_metrics'],
            #     evaluation_obj)

            if 'CLIPScore' in config['evaluation_metrics']:
                clip_score = evaluation_results[img_name][label]['scores']['CLIPScore']
            else:
                clip_score = DEFAULT_CLIP_SCORE
            if 'CLAPScore' in config['evaluation_metrics']:
                clap_score = evaluation_results[img_name][label]['scores']['CLAPScore']
            else:
                clip_score = DEFAULT_CLAP_SCORE
            if config['calc_fluency'] and 'fluency' in config['evaluation_metrics']:
                # fluency_score = perplexities[img_name][label]
                fluency_score = evaluation_results[img_name][label]['scores']['fluency']
            else:
                fluency_score = DEFAULT_PERPLEXITY_SCORE
            if 'style_classification' in config['evaluation_metrics']:
                style_cls_score = evaluation_results[img_name][label]['scores']['style_classification']
            elif 'style_classification_roberta' in config['evaluation_metrics']:
                style_cls_score = evaluation_results[img_name][label]['scores']['style_classification_roberta']
            elif 'style_classification_emoji' in config['evaluation_metrics']:
                style_cls_emoji_score = evaluation_results[img_name][label]['scores']['style_classification_emoji']
                # if type(style_cls_emoji_score)==list:
                try:
                    style_cls_emoji_score = float(style_cls_emoji_score[0].numpy())
                except:
                    style_cls_emoji_score = style_cls_emoji_score[0].item()
                style_cls_score = style_cls_emoji_score
            else:
                style_cls_score = DEFAULT_STYLE_CLS_SCORE

            if not config.get('use_style_model',False): #todo
                if config.get('use_audio_model', False):
                    style_cls_score = clap_score
                    style_cls_scores = clap_scores


            avg_total_score = calculate_avg_score(clip_score, fluency_score, style_cls_score)
            # avg_total_score = calculate_avg_score(clip_score, fluency_score, style_cls_score)#todo: remove

            #todo: I removed the option that there is no style cls  in eval metrics
            # if 'style_classification' in evaluation_results[img_name][label]['scores']:
            #     style_cls_score = evaluation_results[img_name][label]['scores']['style_classification']
            #     style_cls_scores.append(style_cls_score)
            #     avg_total_score = calculate_avg_score(clip_score, fluency_score, style_cls_score)
            # else:
            #     style_cls_score = 'None'
            #     avg_total_score = calculate_avg_score(clip_score, fluency_score)

            clip_scores.append(clip_score)
            clap_scores.append(clap_score)
            fluency_scores.append(fluency_score)
            # style_cls_scores.append(style_cls_score)
            if 'style_classification_emoji' in config['evaluation_metrics']:
                style_cls_emoji_scores.append(style_cls_emoji_score)
                style_cls_scores.append(style_cls_emoji_score)
            else:
                style_cls_scores.append(style_cls_score)

            avg_total_scores.append(avg_total_score)
            res_text = evaluation_results[img_name][label]['res']
            gt_text = evaluation_results[img_name][label]['gt']

            total_res_text.append(res_text)
            total_gt_text.append(gt_text)
            total_captions.append(Caption(img_name, label, res_text, gt_text, evaluation_results[img_name]['img_path'],
                                          label, clip_score, fluency_score, avg_total_score, gts_data[img_name]['factual'],
                                          style_cls_score, style_cls_emoji_scores,clap_scores))

    clip_scores_table = get_table_for_wandb(clip_scores)
    fluency_scores_table = get_table_for_wandb(fluency_scores)
    style_cls_scores_table = get_table_for_wandb(style_cls_scores)
    # style_cls_emoji_scores_table = get_table_for_wandb(style_cls_emoji_scores)
    avg_total_scores_table = get_table_for_wandb(avg_total_scores)
    # if 'style_classification' in evaluation_results[img_name][label]['scores']:
    #     style_cls_scores_table = get_table_for_wandb(style_cls_scores)

    if config['wandb_mode'] == 'online':
        wandb.log({'details_evaluation/style_cls_score': style_cls_scores_table,
                   # 'details_evaluation/style_cls_emoji_score': style_cls_emoji_scores_table,
                   'details_evaluation/clip_score': clip_scores_table,
                   'details_evaluation/fluency_score': fluency_scores_table,
                   'details_evaluation/avg_total_score': avg_total_scores_table})

    total_score_and_text = pd.concat(
        [pd.DataFrame({'avg_total_score': avg_total_scores}, index=list(range(len(avg_total_scores)))),
         pd.DataFrame({'total_res_text': total_res_text}, index=list(range(len(total_res_text)))),
         pd.DataFrame({'total_gt_text': total_gt_text}, index=list(range(len(total_gt_text))))], axis=1)
    if config['wandb_mode'] == 'online':
        wandb.log({'details_evaluation/total_score_text': total_score_and_text})

    avg_clip_score = np.mean(clip_scores) / config['desired_min_CLIP_score']
    avg_fluency_score = np.mean(fluency_scores)
    avg_style_cls_score = np.mean(style_cls_scores)
    # avg_style_cls_emoji_score = torch.mean(torch.stack(style_cls_emoji_scores))
    if 'requires_min_fluency_score' in config:
        if avg_fluency_score > config['requires_min_fluency_score']:
            final_avg_total_score = calculate_avg_score_wo_fluence(avg_clip_score, avg_style_cls_score)
        else:
            final_avg_total_score = 0
    else:
        final_avg_total_score = calculate_avg_score(avg_clip_score, avg_fluency_score,avg_style_cls_score)
    final_avg_total_score_clip_gpt = calculate_avg_score_clip_gpt(avg_clip_score, avg_fluency_score)

    # if style_cls_score != 'None':
    #     avg_style_cls_score = np.mean(style_cls_scores)
    #     final_avg_total_score = calculate_avg_score(avg_clip_score, avg_fluency_score, avg_style_cls_score)
    # else:
    #     avg_style_cls_score = 0
    #     final_avg_total_score = calculate_avg_score(avg_clip_score, avg_fluency_score)

    print("*****************************")
    print("*****************************")
    print("*****************************")
    print(f'style_cls_scores={style_cls_scores}, avg_style_cls_score={avg_style_cls_score}')
    # print(f'style_cls_emoji_scores={style_cls_emoji_scores}, avg_style_cls_score={avg_style_cls_emoji_score}')
    # print(f'avg_style_cls_score={avg_style_cls_emoji_score}')
    print(f'clip_scores={clip_scores}, avg_clip_score={avg_clip_score}'
          f'\nfluency_scores={fluency_scores}, avg_fluency_score={avg_fluency_score}')
    print(f'final_avg_total_score={final_avg_total_score}')
    print(f'final_avg_total_score_clip_gpt={final_avg_total_score_clip_gpt}')
    print("*****************************")
    print("*****************************")
    print("*****************************")
    if config['wandb_mode'] == 'online':
        wandb.log({'evaluation/mean_style_cls_scores': avg_style_cls_score,
                   # 'evaluation/mean_style_cls_emoji_scores': avg_style_cls_emoji_score,
                   'evaluation/avg_clip_score': avg_clip_score,
                   'evaluation/mean_fluency_scores': avg_fluency_score,
                   'evaluation/final_avg_total_score': final_avg_total_score,
                   'evaluation/final_avg_total_score_clip_gpt': final_avg_total_score_clip_gpt})

    write_evaluation_results(total_captions, final_avg_total_score, results_dir, config)
    print('Finish to evaluate results!')


def get_desired_style_embedding_vector_and_std(config, label, mean_embedding_vectors, std_embedding_vectors = None):
    if config['use_style_model'] and (config['style_type'] == 'emoji' or config['style_type'] == 'style_embed'):
        if config['style_type'] == 'emoji':
            if config['use_single_emoji_style']:
                desired_style_embedding_vector = torch.nn.functional.one_hot(torch.tensor(config['desired_labels'].index(label)), num_classes=len(config['desired_labels']))+EPSILON
            else:  # use several emojis
                desired_style_embedding_vector = torch.tensor(np.ones(config['num_classes']) * EPSILON)
                for i in range(config['num_classes']):
                    if i in config['idx_emoji_style_dict'][label]:
                        desired_style_embedding_vector[i] = 1
                # desired_style_embedding_vector = torch.nn.functional.one_hot(torch.tensor(config['idx_emoji_style_dict'][label]), num_classes=config['num_classes'])+EPSILON
                desired_style_embedding_vector = torch.tensor(
                    desired_style_embedding_vector / torch.sum(desired_style_embedding_vector))
            desired_style_embedding_vector_std = None
        elif config['style_type'] == 'style_embed':
            desired_style_embedding_vector = mean_embedding_vectors[label]
            # desired_style_embedding_vector_std = config['embedding_vectors_std'][label]
            # real std
            if std_embedding_vectors:
                desired_style_embedding_vector_std = std_embedding_vectors[label]
            else:
                desired_style_embedding_vector_std = config['embedding_vectors_std'][label]
            # if label == 'positive':
            #     desired_style_embedding_vector_std = config['std_embedding_vectors_positive']
            # elif label == 'negative':
            #     desired_style_embedding_vector_std = config['std_embedding_vectors_negative']
    else:
        desired_style_embedding_vector = None;
        desired_style_embedding_vector_std = None
    return desired_style_embedding_vector, desired_style_embedding_vector_std


def initial_variables():
    def get_desired_labels(config, mean_embedding_vec_path, std_embedding_vec_path):
        if config['use_style_model'] and config['style_type'] == 'style_embed':
            with open(mean_embedding_vec_path, 'rb') as fp:
                mean_embedding_vectors = pickle.load(fp)
            # desired_labels_list = list(mean_embedding_vectors.keys())
            #load_std_vec_embed
            with open(std_embedding_vec_path, 'rb') as fp:
                std_embedding_vectors = pickle.load(fp)
        else:
            mean_embedding_vectors = None
            std_embedding_vectors = None
        desired_labels_list = config['desired_labels']
        if config['imitate_text_style']:
            desired_labels_list = config['text_to_imitate_list']
        # if config['debug']:
        #     desired_labels_list = [desired_labels_list[0]]
        return desired_labels_list, mean_embedding_vectors, std_embedding_vectors

    args = get_args()
    config = get_hparams(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_idx_num']
    data_dir = os.path.join(os.path.expanduser('~'), 'data')
    mean_embedding_vec_path = os.path.join(os.path.expanduser('~'), config['mean_vec_emb_file'])
    std_embedding_vec_path = os.path.join(os.path.expanduser('~'), config['std_vec_emb_file'])

    if config['debug_mac']:
        config['debug'] = True

    if config['debug']:
        config['max_num_of_imgs'] = 1
        config['target_seq_length'] = 2
        config['beam_size'] = 2
        config['wandb_mode'] = 'disabled'
        # config['calc_fluency'] = False


    imgs_to_test = get_list_of_imgs_for_caption(config)
    if config.get("use_img_path",False):
        imgs_to_test = [config["use_img_path"]]

    if not config['use_style_model']:
        config['text_style_scale'] = 0

    if config['factual_captions_path']:
        with open(config['factual_captions_path'], 'rb') as f:
            factual_captions = pickle.load(f)
    else:
        factual_captions = None

    if config['wandb_mode'] == 'online':
        wandb.init(project=config['experiement_global_name'],
                   config=config,
                   resume=config['resume'],
                   id=config['run_id'],
                   mode=config['wandb_mode'],  # disabled, offline, online'
                   tags=config['tags'])

    # handle sweep training names
    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print(f'Current time is: {cur_time}')
    cur_date = datetime.now().strftime("%d_%m_%Y")
    if config['wandb_mode'] == 'online':
        config['training_name'] = f"{wandb.run.name.split('-')[-1]}-{wandb.run.id}-{wandb.run.name}"
    else:
        config['training_name'] = 'tmp'

    experiment_type_dir = os.path.join(os.path.expanduser("~"),'experiments','zero_style_cap',config['dataset'],config['style_type'],config['experiement_global_name'])


    if not os.path.isdir(experiment_type_dir):
        os.makedirs(experiment_type_dir)
    cur_date_dir = os.path.join(experiment_type_dir,cur_date)
    if not os.path.isdir(cur_date_dir):
        os.makedirs(cur_date_dir)
    config['experiment_dir'] = os.path.join(cur_date_dir,config["training_name"])
    results_dir = config['experiment_dir']
    tgt_results_path = os.path.join(results_dir, f'results_{cur_time}.csv')

    if config['wandb_mode'] == 'online':
        wandb.config.update(config, allow_val_change=True)

    txt_cls_model_path = os.path.join(os.path.expanduser('~'), config['txt_cls_model_path'])
    evaluation_obj = {}
    if 'evaluation_metrics' in config:
        if config.get('use_style_threshold', False) or config.get('iterate_until_good_fluency', False):
            if 'style_classification' in config['evaluation_metrics']:
                evaluation_obj['style_classification'] = STYLE_CLS(txt_cls_model_path, config['cuda_idx_num'],
                                                        config['labels_dict_idxs'], data_dir, config[
                                                            'hidden_state_to_take_txt_cls'], config['max_batch_size_style_cls'])
                print(f"style_cls_obj = STYLE_CLS")
            if 'style_classification_roberta' in config['metrics']:
                evaluation_obj['style_classification_roberta'] = STYLE_CLS_ROBERTA(config['finetuned_roberta_config'],
                                                  config['finetuned_roberta_model_path'], config['cuda_idx_num'], config['labels_dict_idxs_roberta'],
                                                  data_dir)
                print(f"style_cls_obj = STYLE_CLS_ROBERTA")

            if 'style_classification_emoji' in config['evaluation_metrics']:
                evaluation_obj['style_classification_emoji'] = STYLE_CLS_EMOJI(config['emoji_vocab_path'], config['maxlen_emoji_sentence'], config['emoji_pretrained_path'], config['idx_emoji_style_dict'])

        if 'fluency' in config['evaluation_metrics'] and config['calc_fluency']:
            evaluation_obj['fluency'] = Fluency(config['desired_labels'])

    desired_labels_list, mean_embedding_vectors, std_embedding_vectors = get_desired_labels(config, mean_embedding_vec_path, std_embedding_vec_path)
    # if config['debug']:
    #     config['desired_labels'] = [config['desired_labels'][0]]

    print(f'saving experiement outputs in {os.path.abspath(config["experiment_dir"])}')

    if not os.path.exists(config['experiment_dir']):
        os.makedirs(config['experiment_dir'])
    print('------------------------------------------------------------------------------------------------------')
    print('Training config:')
    for k, v in config.items():
        print(f'{k}: {v}')
    print('------------------------------------------------------------------------------------------------------')
    with open(os.path.join(config['experiment_dir'], 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    with open(os.path.join(config['experiment_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    if config['save_config_file']:
        print('saved experiment config in ', os.path.join(config['experiment_dir'], 'config.pkl'))

    img_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ""))))
    img_dict_img_arithmetic = defaultdict(lambda: defaultdict(lambda: ""))  # img_path,dataset_type
    tmp_text_loss = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: "")))
    debug_tracking = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))  # word_num:iteration:module:list
    model_path = os.path.join(os.path.expanduser('~'), config['best_model_name'])

    config['requires_min_clip_score_val']['positive'] = config['requires_min_clip_score_val_pos']
    config['requires_min_clip_score_val']['negative'] = config['requires_min_clip_score_val_neg']
    config['requires_num_min_clip_score_val']['positive'] = config['requires_num_min_clip_score_val_pos']
    config['requires_num_min_clip_score_val']['negative'] = config['requires_num_min_clip_score_val_neg']
    config['requires_min_style_score']['positive'] = config['requires_min_style_score_pos']
    config['requires_min_style_score']['negative'] = config['requires_min_style_score_neg']

    return config, data_dir, results_dir, model_path, txt_cls_model_path, mean_embedding_vec_path, tgt_results_path,\
           cur_time, img_dict, img_dict_img_arithmetic, debug_tracking,tmp_text_loss, factual_captions,\
           desired_labels_list, mean_embedding_vectors ,std_embedding_vectors, imgs_to_test, evaluation_obj


def main():
    config, data_dir, results_dir, model_path, txt_cls_model_path, mean_embedding_vec_path, tgt_results_path, cur_time,\
    img_dict, img_dict_img_arithmetic, debug_tracking, tmp_text_loss, factual_captions, desired_labels_list,\
    mean_embedding_vectors, std_embedding_vectors, imgs_to_test, evaluation_obj = initial_variables()

    gts_data = get_gts_data(config['annotations_path'], config['imgs_path'], config['data_type'], factual_captions, config['max_num_imgs2test'])
    if not config['debug_mac']:
        text_generator = CLIPTextGenerator(cuda_idx=config['cuda_idx_num'], model_path=model_path,
                                           tmp_text_loss=tmp_text_loss, config=config, evaluation_obj=evaluation_obj,
                                           **config)
    else:
        text_generator = None

    # go over all images
    evaluation_results = {}  # total_results_structure
    if config['reverse_imgs_list']:
        imgs_to_test.reverse()
    failed_img_names = []
    for img_path_idx, img_path in enumerate(imgs_to_test):
        if len(evaluation_results) >= config['max_num_imgs2test'] and config['max_num_imgs2test'] > 0:
            break
        if img_path_idx < config['img_idx_to_start_from']:
            continue
        print(f"Img num = {img_path_idx}")
        img_name = img_path.split('/')[-1].split('.')[0]
        if config["dataset"] == "senticap":
            try:
                img_name = int(img_name)
            except:
                pass
        config['img_path'] = img_path
        config['img_path_idx'] = img_path_idx
        evaluation_results[img_name] = {'img_path': img_path}
        if not os.path.isfile(config['img_path']):
            continue
        # go over all labels
        for label_idx, label in enumerate(desired_labels_list):
            if config['wandb_mode'] == 'online':
                wandb.log({'test/img_idx': img_path_idx, 'img_name': img_name, 'style': label})
            config['cond_text'] = config["cond_text_dict"][label]
            config = update_running_params(label, config)
            if not config['debug_mac']:
                text_generator.update_config(config)
                # text_generator = CLIPTextGenerator(cuda_idx=config['cuda_idx_num'], model_path=model_path,
                #                                    tmp_text_loss=tmp_text_loss, config=config,
                #                                    evaluation_obj=evaluation_obj,
                #                                    **config)
                clip_img = None
                if config.get('update_ViT',False):
                    image_features, clip_img = text_generator.get_img_feature([img_path], None, return_k_v=False,
                                                                              get_preroccessed_img=True,
                                                                              kv_only_first_layer=config.get(
                                                                                  'kv_only_first_layer', True))
                else:
                    image_features = text_generator.get_img_feature([img_path], None, return_k_v=False,
                                                                    get_preroccessed_img=True)
                text_generator.set_params(config['ce_scale'], config['clip_scale'], config['text_style_scale'],
                                          config['beam_size'], config['num_iterations'])
            else:
                image_features = None

            evaluation_results[img_name][label] = {}
            if not config['imitate_text_style']:
                desired_style_embedding_vector, desired_style_embedding_vector_std = get_desired_style_embedding_vector_and_std(config, label, mean_embedding_vectors, std_embedding_vectors)
            if config['debug_mac']:
                evaluation_results[img_name][label] = {'res': 'bla'}
                continue
            print(f"Img num = {img_path_idx}")
            # prompt manipulation or using text style model
            if True:
            # try:
                if config['run_type'] == 'caption':
                    title2print = get_title2print(config['img_path'], config['dataset'], label, config)
                    print(title2print)
                    best_caption = run(config, config['img_path'], desired_style_embedding_vector,
                                       desired_style_embedding_vector_std,
                                       config['cuda_idx_num'], title2print, model_path, config['style_type'], tmp_text_loss,
                                       label, img_dict, debug_tracking, text_generator, image_features, evaluation_obj, clip_img)
                    write_caption_results(img_dict, results_dir, tgt_results_path)
                    # write_results_of_text_style_all_models(img_dict, desired_labels_list,
                    #                                    results_dir, 1, tgt_results_path)
                    if config['write_debug_tracking_file']:
                        write_debug_tracking(results_dir, debug_tracking)
                    # if 'fluency' in config['evaluation_metrics']:
                        # evaluation_obj['fluency'].add_test(best_caption, img_name, label)

                # image manipulation
                elif config['run_type'] == 'arithmetics':
                    config['arithmetics_weights'] = [float(x) for x in config['arithmetics_weights']]
                    factual_img_style = get_full_path_of_stylized_images(data_dir, config["style_img"]["factual"])
                    img_style = get_full_path_of_stylized_images(data_dir, config["style_img"][label])
                    config['arithmetics_imgs'] = [config['img_path'], factual_img_style, img_style]
                    title2print = get_title2print(config['img_path'], config['dataset'],
                                                  label, config)
                    print(title2print)
                    best_caption = run_arithmetic(text_generator, config, model_path, img_dict_img_arithmetic, img_name,
                                                  label, imgs_path=config['arithmetics_imgs'],
                                                  img_weights=config['arithmetics_weights'],
                                                  cuda_idx=config['cuda_idx_num'], title2print=title2print, img_name=img_name, style=label)
                    write_results_image_manipulation(img_dict_img_arithmetic, results_dir, tgt_results_path)
                else:
                    raise Exception('run_type must be caption or arithmetics!')
                evaluation_results[img_name][label]['res'] = best_caption
                # evaluation_results[img_name][label]['gt'] = gts_data[img_name][label]  # todo: handle for flickr
                evaluation_results[img_name][label]['gt'] = None  # todo: handle style type
                evaluation_results[img_name][label]['scores'] = {}
                # evaluation_obj['fluency'].add_results(evaluation_results)
                evaluation_obj = get_evaluation_obj(config, text_generator, evaluation_obj)
                evaluation_results[img_name][label]['scores'] = evaluate_single_res(
                    evaluation_results[img_name][label]['res'], evaluation_results[img_name][label]['gt'],
                    evaluation_results[img_name]['img_path'], label, config['evaluation_metrics'],
                    evaluation_obj)
                perplexities, mean_perplexity = evaluation_obj['fluency'].compute_score_for_single_text(best_caption)
                evaluation_results[img_name][label]['scores']['fluency'] = mean_perplexity
                print(f"evaluation scores: CLIPScore={evaluation_results[img_name][label]['scores']['CLIPScore']}, fluency={evaluation_results[img_name][label]['scores']['fluency']}, ")
                if "style_classification_roberta" in config["evaluation_metrics"]:
                    print(f"style_classification_roberta={evaluation_results[img_name][label]['scores']['style_classification_roberta']}")
                elif "style_classification_emoji" in config["evaluation_metrics"]:
                    print(f"style_classification_emoji={evaluation_results[img_name][label]['scores']['style_classification_emoji']}")
                if "CLIPScore" in config["evaluation_metrics"]:
                    print(f"CLIPScore={evaluation_results[img_name][label]['scores']['CLIPScore']}")
                if "CLAPScore" in config["evaluation_metrics"]:
                    print(f"CLAPScore={evaluation_results[img_name][label]['scores']['CLAPScore']}")

            # except:
            #     print("check why it failed")
            #     failed_img_names.append(img_name)
    evaluate_results(config, evaluation_results, gts_data, results_dir, factual_captions, text_generator,evaluation_obj)
    print("images that failed:")
    print(failed_img_names)
    print('Finish of program!')


if __name__ == "__main__":
    main()
