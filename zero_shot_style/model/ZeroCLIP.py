# import pdb
import csv
import math
import os.path
import heapq
# import pdb
import matplotlib.pyplot as plt
from transformers import AutoConfig
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt_neo import GPTNeoForCausalLM
import torch
import clip
from clip import *
from PIL import Image
from datetime import datetime
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer # SENTIMENT
from transformers import BertTokenizer #TEXT_STYLE
from transformers import AutoModelForCausalLM #gpt-J
#import cv2
from transformers import BertModel
from torch.optim import Adam, SGD
# from zero_shot_style.model.text_style_embedding import TextStyleEmbed
from zero_shot_style.model.text_style_embedding_senticap import TextStyleEmbed
from zero_shot_style.model.text_style_embedding_senticap_based_on_clip import TextStyleEmbedCLIP
# from zero_shot_style.evaluation.evaluation_all import STYLE_CLS
import pickle

import json
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from evaluate import load

factor_clip_style=1000
TOP_SZIE = 512 #200
max_prob_len = -1 #500
MAX_PERPLEXITY = 6000
DEBUG_NUM_WORDS = 10
EPSILON = 0.0000000001

def write_tmp_text_loss(tmp_text_loss):
    def write_results_of_text_style_all_models(img_dict, labels, reults_dir, scales_len, tgt_results_path):
        # img_dict[img_path][style_type][text_style_scale][label]
        if not os.path.isdir(reults_dir):
            os.makedirs(reults_dir)
        print(f'Writing results into: {tgt_results_path}')
        with open(tgt_results_path, 'w') as results_file:
            writer = csv.writer(results_file)
            for img in img_dict.keys():
                img_num_str = img.split('/')[-1].split('.j')[0]
                titles0 = ['img_num']
                titles0.extend([img_num_str] * scales_len * len(labels))
                writer.writerow(titles0)
                titles1 = ['label']
                for label in labels:
                    titles1.extend([label] * scales_len)
                writer.writerow(titles1)
                titles2 = ['model/scale']
                titles2.extend(list(img_dict[img][list(img_dict[img].keys())[0]].keys()) * len(labels))
                writer.writerow(titles2)
                for model_name in img_dict[img]:
                    cur_row = [model_name]
                    for scale in img_dict[img][model_name].keys():
                        for label in img_dict[img][model_name][scale].keys():
                            cur_row.append(img_dict[img][model_name][scale][label])
                    writer.writerow(cur_row)


def log_info(text, verbose=True):
    if verbose:
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string} | {text}')
        sys.stdout.flush()


def add_context(x, y):
    return (x[0] + y[0], x[1] + y[1])

def add_context_clip(x, y):
    return (x[0] + y[0])

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


class CLIPTextGenerator:
    def __init__(self,
                 seed=0,
                 lm_model='gpt-2',
                 # forbidden_tokens_file_path='./forbidden_tokens.npy',
                 # clip_checkpoints='./clip_checkpoints',
                 forbidden_tokens_file_path=os.path.join(os.path.expanduser('~'),'projects/zero-shot-style/zero_shot_style','forbidden_tokens.npy'),
                 clip_checkpoints=os.path.join(os.path.expanduser('~'),'checkpoints','clip_checkpoints'),
                 target_seq_length=15,
                 reset_context_delta=True,
                 num_iterations=5,
                 clip_loss_temperature=0.01,
                 text_style_loss_temperature = 0.0002,
                 clip_scale=1.,
                 ce_scale=0.2,
                 text_style_scale = 1,
                 stepsize=0.3,#todo
                 grad_norm_factor=0.9,
                 fusion_factor=0.99,
                 repetition_penalty=1.,
                 end_token='.',
                 end_factor=1.04, #16.3.23 change from 1.01 to 1.04
                 forbidden_factor=20,
                 cuda_idx = 0,
                 model_path=None,
                 tmp_text_loss=None,
                 use_style_model=False,
                 config=None,
                 model_based_on='bert',
                 evaluation_obj=None,
                 desired_style_bin=None,
                 use_text_style_cutting=False,
                 img_path=None,
                 **kwargs):

        self.style_type = None
        self.img_path = img_path
        self.config = config
        self.use_text_style_cutting = use_text_style_cutting
        if evaluation_obj:
            self.evaluation_obj = evaluation_obj
        if config:
            self.model_based_on = config['model_based_on']
        else:
            self.model_based_on = model_based_on
        if 'iterate_until_good_fluency' in self.config:
            if self.config['iterate_until_good_fluency']:
                print("iterate_until_good_fluency...")
                self.perplexity = load("perplexity", module_type="measurement")

        self.debug_tracking = {} # debug_tracking: debug_tracking[word_num][iteration][module]:<list>
        self.tmp_text_loss = tmp_text_loss
        self.cuda_idx = cuda_idx
        #self.device = f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu"
        self.device = f"cuda" if torch.cuda.is_available() else "cpu"#todo: change
        # self.LM_loss_scale = LM_loss_scale
        # self.CLIP_loss_scale = CLIP_loss_scale
        # self.STYLE_loss_scale = STYLE_loss_scale

        # set Random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize Language model
        self.context_prefix = ''

        if lm_model == 'gpt-neo':
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
            self.lm_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M', output_hidden_states=True)
        elif lm_model == 'gpt-2':
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium') #345M parameters
            self.lm_model = GPT2LMHeadModel.from_pretrained('gpt2-medium', output_hidden_states=True)
            self.context_prefix = self.lm_tokenizer.bos_token
        elif lm_model == 'gpt-j':
            self.lm_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", output_hidden_states=True)
            self.lm_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

            self.context_prefix = self.lm_tokenizer.bos_token
        print(f'lm_model={lm_model}')
        self.lm_model.to(self.device)
        self.lm_model.eval()

        self.forbidden_tokens = np.load(forbidden_tokens_file_path)
        self.capital_letter_tokens = [self.lm_tokenizer.encoder[x] for x in self.lm_tokenizer.encoder.keys() if
                                      (x[0] == 'Ġ' and len(x) > 1 and x[1].isupper())]

        # Freeze LM weights
        for param in self.lm_model.parameters():
            param.requires_grad = False

        # Initialize CLIP
        # self.clip, self.clip_preprocess = clip.load(os.path.join(os.path.expanduser('~'),'projects/zero-shot-style/zero_shot_style','ViT-B/32'), device=self.device,download_root=clip_checkpoints, jit=False)
        self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=self.device,
                                                    download_root=clip_checkpoints, jit=False) #todo

        # self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=self.device,
        #                                             download_root=clip_checkpoints, jit=False,
        #                                             use_flash_attention=False)  # todo


        # convert_models_to_fp32(self.clip)
        self.clip.eval()

        # Init arguments
        self.target_seq_length = int(target_seq_length)
        self.reset_context_delta = reset_context_delta
        self.num_iterations = int(num_iterations)
        self.clip_loss_temperature = clip_loss_temperature
        self.text_style_loss_temperature = text_style_loss_temperature
        self.clip_scale = clip_scale
        self.clip_scale = clip_scale
        self.ce_scale = ce_scale
        self.text_style_scale = text_style_scale
        self.stepsize = stepsize
        self.grad_norm_factor = grad_norm_factor
        self.fusion_factor = fusion_factor
        self.repetition_penalty = repetition_penalty
        self.end_token = self.lm_tokenizer.encode(end_token)[0]
        self.end_factor = end_factor
        self.ef_idx = 1
        self.forbidden_factor = forbidden_factor

        # SENTIMENT: adding the sentiment model
        # task = 'sentiment'
        # MODEL = f"cardiffnlp/twitter-roberta-base-{task}-latest"
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

        self.sentiment_temperature = config.get('sentiment_temperature', 0.01)
        self.sentiment_model_name = MODEL
        if config['style_type'] == 'roberta':
            # tokenizer = AutoTokenizer.from_pretrained(MODEL)
            # config = AutoConfig.from_pretrained(MODEL)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
            # f_roberta_config = AutoConfig.from_pretrained(self.config['finetuned_roberta_config'])
            # self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.config['finetuned_roberta_model_path'],
            #                                                                       config=f_roberta_config)
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()

            # SENTIMENT: Freeze sentiment model weights
            for param in self.sentiment_model.parameters():
                param.requires_grad = False

            # SENTIMENT: tokenizer for sentiment analysis module
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)
            # config = AutoConfig.from_pretrained(MODEL)

            # self.sentiment_tokenizer_name = self.sentiment_model_name
            # self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_tokenizer_name)

            # MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
            # sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)

            # SENTIMENT: fields for type and scale of sentiment
            self.sentiment_scale = 1
            # self.sentiment_temperature = config.get('sentiment_temperature',0.01)
            self.sentiment_type = 'none' # SENTIMENT: sentiment_type can be one of ['positive','negative','neutral', 'none']

        self.use_style_model = use_style_model
        if config['style_type'] == 'erc':
            #########use erc model:
            self.text_style_tokenizer_erc = AutoTokenizer.from_pretrained("tae898/emoberta-large")
            self.text_style_erc_model = AutoModelForSequenceClassification.from_pretrained(
                "tae898/emoberta-large", num_labels=7
            )
            self.text_style_erc_model.to(self.device)
            for param in self.text_style_erc_model.parameters():
                param.requires_grad = False

            self.text_style_erc_model.eval()
            #########


        # TorchEmoji: emoji style model
        if config['style_type'] == 'emoji':
            print('Tokenizing using dictionary from {}'.format(config['emoji_vocab_path']))
            with open(config['emoji_vocab_path'], 'r') as f:
                vocabulary = json.load(f)
            self.emoji_st_tokenizer = SentenceTokenizer(vocabulary, config['maxlen_emoji_sentence'])
            print('Loading emoji style model  from {}.'.format(config['emoji_pretrained_path']))
            self.emoji_style_model = torchmoji_emojis(config['emoji_pretrained_path'])
            # self.emoji_style_model.to(self.device)
            # TEXT_STYLE: Freeze text style model weights
            for param in self.emoji_style_model.parameters():
                param.requires_grad = False
            self.emoji_style_model.eval()
            self.check_if_cut_score = True
            # self.check_if_cut_score = {}
            # for idx_p in range(self.beam_size):
            #     self.check_if_cut_score[idx_p] = True


        # TEXT STYLE: adding the text style model
        elif config['style_type'] == 'style_embed':
            data_dir = os.path.join(os.path.expanduser('~'),'data')
            if self.use_text_style_cutting:
                self.text_style_cls = STYLE_CLS(config['txt_cls_model_paths'], data_dir, self.cuda_idx, config['labels_dict_idxs'],
                                 config['hidden_state_to_take_txt_cls'])
            else:
                if self.model_based_on == 'bert':
                    self.text_style_model = TextStyleEmbed(device=self.device, hidden_state_to_take=config['hidden_state_to_take_txt_style_embedding'])
                elif self.model_based_on == 'clip':
                    self.text_style_model = TextStyleEmbedCLIP(device=self.device)

                if 'cpu' in self.device:
                    checkpoint = torch.load(config['txt_embed_model_paths'], map_location=torch.device('cpu'))
                else:
                    checkpoint = torch.load(config['txt_embed_model_paths'], map_location='cuda:0')

                self.text_style_model.load_state_dict(checkpoint['model_state_dict'])
                self.text_style_model.to(self.device)
                self.text_style_model.eval()

            ##############
            self.use_text_style = True
            #self.text_style_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_style_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.text_style_model_name = model_path

            if self.use_style_model and self.style_type != 'erc':
                print(f"Loading embedding style model from: {self.text_style_model_name}")
                if self.model_based_on == 'bert':
                    self.text_style_model = TextStyleEmbed(device=self.device, hidden_state_to_take=config['hidden_state_to_take_txt_style_embedding'])
                elif self.model_based_on == 'clip':
                    self.text_style_model = TextStyleEmbedCLIP(device=self.device)

                if 'cpu' in self.device:
                    checkpoint = torch.load(self.text_style_model_name, map_location=torch.device('cpu'))
                else:
                    checkpoint = torch.load(self.text_style_model_name, map_location='cuda:0')

                self.text_style_model.load_state_dict(checkpoint['model_state_dict'])
                self.text_style_model.to(self.device)
                self.text_style_model.eval()

                # TEXT_STYLE: Freeze text style model weights
                for param in self.text_style_model.parameters():
                    param.requires_grad = False

            self.desired_style_embedding_vector = ''
            self.desired_style_embedding_std_vector = ''
            # TEXT_STYLE: tokenizer for text style analysis module
            #self.text_style_tokenizer_name = self.text_style_model_name
            #self.text_style_tokenizer = AutoTokenizer.from_pretrained(self.text_style_tokenizer_name)

    def update_config(self,config):
        self.config = config
        self.img_path = config['img_path']
        self.img_idx = config['img_path_idx']

    def set_params(self, ce_scale, clip_scale, text_style_scale, beam_size,  num_iterations):
        self.ce_scale = ce_scale
        self.clip_scale = clip_scale
        self.text_style_scale = text_style_scale
        self.beam_size = beam_size
        self.num_iterations = num_iterations

    def get_debug_tracking(self):
        return self.debug_tracking


    def get_img_feature(self, img_path, weights, source_clip = False, use_flash_attention = False, k=None, v=None, return_k_v=False, get_preroccessed_img=False,kv_only_first_layer=True):
        #imgs = [Image.fromarray(cv2.imread(x)) for x in img_path]
        #imgs = [Image.fromarray(cv2.imread(x).astype('uint8'), 'RGB') for x in img_path]
        #imgs = [Image.fromarray(cv2.imread(x), 'RGB') for x in img_path]
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]
        clip_img = clip_imgs[0] #todo:handle to several images
        if self.config.get('update_ViT',False):
            if self.model_based_on == 'bert' or source_clip:
                # image_fts = [self.clip.encode_image(x,return_k_v=return_k_v) for x in clip_imgs]
                image_fts = []
                for x in clip_imgs:
                    if return_k_v:
                        image_fts_s, k, v = self.clip.encode_image(x, return_k_v=return_k_v,kv_only_first_layer=kv_only_first_layer)
                    else:
                        image_fts_s = self.clip.encode_image(x, return_k_v=return_k_v,kv_only_first_layer=kv_only_first_layer)
                    if type(image_fts_s) == tuple:
                        image_fts_s = image_fts_s[0]
                    image_fts.append(image_fts_s)
                    # self.k_clip = k
                    # self.v_clip = v
                # image_fts = [self.clip.encode_image(x,return_k_v=return_k_v) for x in clip_imgs]
            elif self.model_based_on == 'clip':  # for text_style
                image_fts = [self.text_style_model.forward_im(x) for x in clip_imgs]
            if weights is not None:
                image_features = sum([x * weights[i] for i, x in enumerate(image_fts)])
            else:
                image_features = sum(image_fts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if return_k_v:
                return image_features, k, v, clip_img
            else:
                if get_preroccessed_img:
                    return image_features, clip_img
                else:
                    return image_features
        else:
            with torch.no_grad():
                if self.model_based_on == 'bert' or source_clip:
                    image_fts = [self.clip.encode_image(x) for x in clip_imgs]
                    if type(image_fts[0]) == tuple:
                        image_fts[0] = image_fts[0][0]
                elif self.model_based_on == 'clip': #for text_style
                    image_fts = [self.text_style_model.forward_im(x) for x in clip_imgs]

                if weights is not None:
                    image_features = sum([x * weights[i] for i, x in enumerate(image_fts)])
                else:
                    image_features = sum(image_fts)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.detach()

    def get_txt_features(self, text, source_clip = False):
        with torch.no_grad():
            if self.model_based_on == 'bert' or source_clip:
                clip_texts = clip.tokenize(text).to(self.device)
                text_features = self.clip.encode_text(clip_texts)
            elif self.model_based_on == 'clip':  # for text_style
                text_features = self.text_style_model(text)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach()

    def get_combined_feature(self, img_path, texts, weights_i, weights_t):
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]
        clip_texts = [clip.tokenize(x).to(self.device) for x in texts]

        with torch.no_grad():
            if self.model_based_on == 'bert':
                image_fts = [self.clip.encode_image(x) for x in clip_imgs]
                text_fts = [self.clip.encode_text(x) for x in clip_texts]
            elif self.model_based_on == 'clip': #for text_style
                image_fts = [self.text_style_model.forward_im(x) for x in clip_imgs]
                text_fts = [self.text_style_model(x) for x in texts]


            features = sum([x * weights_i[i] for i, x in enumerate(image_fts)])
            if weights_t is not None:
                features += sum([x * weights_t[i] for i, x in enumerate(text_fts)])

            features = features / features.norm(dim=-1, keepdim=True)
            return features.detach()

    def run(self, image_features, cond_text, beam_size, text_style_scale = None, text_to_imitate = None, desired_style_embedding_vector = None, desired_style_embedding_std_vector = None, style_type = None,img_idx=None, img_name=None, style=None, desired_style_bin=False,clip_img=None):
    
        # SENTIMENT: sentiment_type can be one of ['positive','negative','neutral', 'none']
        self.image_features = image_features
        self.src_image_features = image_features.detach()
        self.clip_img= clip_img
        self.text_style_list = text_to_imitate
        self.img_idx = img_idx
        self.img_name = img_name
        self.style = style
        self.desired_style_bin = desired_style_bin
        # self.check_if_cut_score = {}
        # for idx_p in range(self.beam_size):
        #     self.check_if_cut_score[idx_p] = True
        if self.use_style_model:
            self.text_style_scale = text_style_scale
            self.style_type = style_type #'clip','twitter','emotions' , 'erc' or 'roberta'
            if self.style_type == 'style_embed':
                if not text_to_imitate:
                    # self.desired_style_embedding_vector = desired_style_embedding_vector.to(self.device)
                    self.desired_style_embedding_vector = desired_style_embedding_vector
                    self.desired_style_embedding_std_vector = desired_style_embedding_std_vector

                else: #there is text_to_imitate:
                    #use clip features
                    if style_type=='clip':#'clip','twitter','emotions'
                        self.text_style_features = self.get_txt_features(self.text_to_imitate)
                        # use my text style model features
                    else: #style_type=='twitter' or 'emotions'
                        #### based on bert
                        tokenized_text_to_imitate = self.text_style_tokenizer(text_to_imitate, padding='max_length',
                                                                            max_length=TOP_SZIE, truncation=True,
                                                                            return_tensors="pt")
                        masks_mimic = tokenized_text_to_imitate['attention_mask'].to(self.device)
                        input_ids_mimic = tokenized_text_to_imitate['input_ids'].squeeze(1).to(self.device)
                        embedding_of_text_to_imitate = self.text_style_model(input_ids_mimic, masks_mimic) #embeding vector
                        # #### based on clip
                        # embedding_of_text_to_imitate = self.text_style_model(text_to_imitate) #embeding vector
                        embedding_of_text_to_imitate.to(self.device)
                        # self.desired_style_embedding_vector = embedding_of_text_to_imitate.to(self.device)
                        self.desired_style_embedding_vector = embedding_of_text_to_imitate

        context_tokens = self.lm_tokenizer.encode(self.context_prefix + cond_text)

        output_tokens, output_text = self.generate_text(context_tokens, beam_size)
        
        return output_text

    def generate_text(self, context_tokens, beam_size):
        context_tokens = torch.tensor(context_tokens, device=self.device, dtype=torch.long).unsqueeze(0)

        gen_tokens = None
        scores = None
        seq_lengths = torch.ones(beam_size, device=self.device)
        is_stopped = torch.zeros(beam_size, device=self.device, dtype=torch.bool)

        for i in range(self.target_seq_length):
            self.debug_tracking[i] = {}
            probs = self.get_next_probs(i, context_tokens)
            logits = probs.log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                context_tokens = context_tokens.expand(beam_size, *context_tokens.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)

                if gen_tokens is None:
                    gen_tokens = next_tokens
                else:
                    gen_tokens = gen_tokens.expand(beam_size, *gen_tokens.shape[1:])
                    gen_tokens = torch.cat((gen_tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1) # flat all beams to one vector and select the topk
                next_tokens_source = next_tokens // scores_sum.shape[1] # choose the token before end relating to the topk
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                gen_tokens = gen_tokens[next_tokens_source]
                gen_tokens = torch.cat((gen_tokens, next_tokens), dim=-1) #ADD DEBUG
                context_tokens = context_tokens[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            context_tokens = torch.cat((context_tokens, next_tokens), dim=1)
            is_stopped = is_stopped + next_tokens.eq(self.end_token).squeeze()

            ####
            tmp_scores = scores / seq_lengths
            tmp_output_list = gen_tokens.cpu().numpy()
            tmp_output_texts = [
                self.lm_tokenizer.decode(tmp_output)
                for tmp_output, tmp_length in zip(tmp_output_list, seq_lengths)
            ]
            tmp_order = tmp_scores.argsort(descending=True)
            tmp_output_texts = [tmp_output_texts[i] + ' %% ' + str(tmp_scores[i].cpu().numpy()) for i in tmp_order]
            log_info(tmp_output_texts, verbose=True)
            ####

            if is_stopped.all():
                break

        scores = scores / seq_lengths
        output_list = gen_tokens.cpu().numpy()
        output_texts = [
            self.lm_tokenizer.decode(output[: int(length)])
            for output, length in zip(output_list, seq_lengths)
        ]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]

        return context_tokens, output_texts

    def get_next_probs(self, i, context_tokens):
        last_token = context_tokens[:, -1:]
        
        context = None
        if self.reset_context_delta and context_tokens.size(1) > 1:
            context = self.lm_model(context_tokens[:, :-1])["past_key_values"]
        
        # Logits of LM with unshifted context
        logits_before_shift = self.lm_model(context_tokens)["logits"]
        logits_before_shift = logits_before_shift[:, -1, :]
        probs_before_shift = nn.functional.softmax(logits_before_shift, dim=-1)
        
        if context:
            context = self.shift_context(i, context, last_token, context_tokens, probs_before_shift)
        lm_output = self.lm_model(last_token, past_key_values=context)
        logits, past = (
            lm_output["logits"],
            lm_output["past_key_values"],
        )
        logits = logits[:, -1, :]

        logits = self.update_special_tokens_logits(context_tokens, i, logits)

        probs = nn.functional.softmax(logits, dim=-1)
        probs = (probs ** self.fusion_factor) * (probs_before_shift ** (1 - self.fusion_factor))
        probs = probs / probs.sum()

        return probs

    def preprocess_text_for_roberta(self, text):
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

    # SENTIMENT: function we added for changing the result to the requested sentiment
    def get_sentiment_loss(self, probs, context_tokens,sentiment_type):

        top_size = TOP_SZIE
        _, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        sentiment_loss = 0
        losses = []
        style_probs = {}
        for idx_p in range(probs.shape[0]): #go over all beams
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]: #go over all optional topk next word
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))
            # ######todo: daniela debug    effect of update CLIP
            # # top_texts = ["The bedroom used child abuse"]*DEBUG_NUM_WORDS+["The bedroom of a sweet baby"]*DEBUG_NUM_WORDS
            # for i in range(len(top_texts)):
            #     if i<=len(top_texts)/2:
            #         top_texts[i] = "The bedroom used child abuse"
            #     else:
            #         top_texts[i] = "The bedroom of a sweet baby"
            # ######todo: daniela debug    effect of update CLIP
            #get score for text
            with torch.no_grad():
                text_list = self.preprocess_text_for_roberta(top_texts)
                encoded_input = self.sentiment_tokenizer(text_list, padding=True, return_tensors='pt').to(self.device)
                output = self.sentiment_model(**encoded_input)
                scores = output[0].detach()
                scores1 = nn.functional.softmax(scores, dim=-1)
                scores2 = nn.functional.softmax(scores1,dim=0)
                # sentiment_grades = None
                if sentiment_type=='positive':
                    sentiment_grades= scores2[:,2]
                elif sentiment_type=='neutral':
                    sentiment_grades = scores2[:, 1]
                elif sentiment_type=='negative':
                    sentiment_grades = scores2[:, 0]

                # inputs = self.sentiment_tokenizer(top_texts, padding=True, return_tensors="pt")
                # inputs['input_ids'] = inputs['input_ids'].to(self.sentiment_model.device)
                # inputs['attention_mask'] = inputs['attention_mask'].to(self.sentiment_model.device)
                # logits = self.sentiment_model(**inputs)['logits']
                                   
                # # sentiment_grades = None
                # if sentiment_type=='positive':
                #         sentiment_grades= nn.functional.softmax(logits, dim=-1)[:,2]
                # elif sentiment_type=='neutral':
                #         sentiment_grades= nn.functional.softmax(logits, dim=-1)[:,1]
                # elif sentiment_type=='negative':
                #         sentiment_grades= nn.functional.softmax(logits, dim=-1)[:,0]
                sentiment_grades = sentiment_grades.unsqueeze(0)
                
                # predicted_probs = nn.functional.softmax(sentiment_grades / self.clip_loss_temperature, dim=-1).detach()
                predicted_probs = nn.functional.softmax(sentiment_grades / self.sentiment_temperature, dim=-1).detach() #todo: parametrize it
                predicted_probs = predicted_probs.type(torch.float32).to(self.device)
             
            
            target = torch.zeros_like(probs[idx_p], device=self.device)
            target[top_indices[idx_p]] = predicted_probs[0]
            # target = target.unsqueeze(0)
            cur_sentiment_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            # x = np.arange(0,probs.shape[1],1)#top_indices[idx_p]
            # y = target.cpu().numpy()
            # plt.figure()
            # plt.plot(x, y)
            # plt.title(f"style probs for beam_idx={idx_p}")
            # plt.show(block=False)

            sentiment_loss += cur_sentiment_loss
            losses.append(cur_sentiment_loss)

            if self.config.get('update_ViT',False):
                style_probs[idx_p] = target
        
        loss_string = ''
        for idx_p in range(probs.shape[0]): #go over all beams
            if idx_p==0:
                loss_string = f'{losses[0]}'
            else:
                loss_string = loss_string+'%, '+f'{losses[idx_p]}'
        return sentiment_loss, losses, style_probs, target

    def get_text_style_loss_with_clip(self, probs, context_tokens):
        for p_ in self.clip.transformer.parameters():
            if p_.grad is not None:
                p_.grad.data.zero_()

        top_size = TOP_SZIE
        _, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        text_style_loss = 0
        losses = []

        for idx_p in range(probs.shape[0]):  # go over all beams
            top_texts = []
            prefix_text = prefix_texts[idx_p]

            for x in top_indices[idx_p]:  # go over all optional topk next word
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))

            top_text_features = self.get_txt_features(top_texts)

            with torch.no_grad():
                similiraties = (self.text_style_features @ top_text_features.T)
                target_probs = nn.functional.softmax(similiraties / self.clip_loss_temperature, dim=-1).detach()#todo: check if to change it
                target_probs = target_probs.type(torch.float32)
            target = torch.zeros_like(probs[idx_p])
            target[top_indices[idx_p]] = target_probs[0]
            target = target.unsqueeze(0)
            cur_text_style_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            text_style_loss += cur_text_style_loss
            losses.append(cur_text_style_loss)

        loss_string = ''
        for idx_p in range(probs.shape[0]):  # go over all beams
            if idx_p == 0:
                loss_string = f'{losses[0]}'
            else:
                loss_string = loss_string + '%, ' + f'{losses[idx_p]}'
        return text_style_loss, losses

    def get_text_style_loss_erc(self, probs, context_tokens):
        #use representative vector for calculating the distance between candidates and the representative vecotr
        top_size = TOP_SZIE
        top_probs_LM, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        text_style_loss = 0
        losses = []
        best_sentences = []
        debug_best_top_texts_style = []
        debug_best_probs_vals_style = []
        if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
            print("in text_style loss:")
        for idx_p in range(probs.shape[0]):  # go over all beams
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]:  # go over all optional topk next word
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))

            with torch.no_grad():
                # #########to define:
                # self.text_style_tokenizer_erc = AutoTokenizer.from_pretrained("tae898/emoberta-large")
                # self.text_style_erc_model = AutoModelForSequenceClassification.from_pretrained(
                #     "tae898/emoberta-large", num_labels=7
                # )
                # for param in self.text_style_erc_model.parameters():
                #     param.requires_grad = False
                #
                # self.text_style_erc_model.eval()
                #
                # #########
                inputs = self.text_style_tokenizer_erc(top_texts, padding=True, return_tensors="pt")
                inputs['input_ids'] = inputs['input_ids'].to(self.device)
                inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
                outputs = self.text_style_erc_model(
                    **{"input_ids": inputs['input_ids'], "attention_mask": inputs['attention_mask']},
                    output_attentions=True,
                    output_hidden_states=True,
                ) #outputs.logits_classes: "neutral","joy","surprise","anger","sadness","disgust","fear"
                pos_prob_val = torch.unsqueeze(outputs.logits[:, 1], dim=0)
                # neg_prob_val = torch.unsqueeze(outputs.logits[:, [3, 4, 5, 6]].sum(dim=-1), dim=0)
                neg_prob_val = torch.unsqueeze(torch.max(outputs.logits[:, [3, 4, 5, 6]],dim=-1).values, dim=0)
                pos_neg_vec = torch.cat((pos_prob_val, neg_prob_val))
                style2id = {'positive':0, 'negative':1}

                pos_neg_prob_vec = torch.nn.functional.softmax(pos_neg_vec, dim=0) #create vector probability between pos and neg scores

                style_score_vec = pos_neg_prob_vec[style2id[self.style]]
                predicted_probs = torch.nn.functional.softmax(style_score_vec).detach() #create vector probability between all candidates of top_text
                # print(f"prob_vec = {predicted_probs}")
                # predicted_probs = nn.functional.softmax(text_style_grades / self.text_style_loss_temperature, dim=-1).detach()
                predicted_probs = predicted_probs.type(torch.float32).to(self.device)

            #todo: debug
            # val_top_predicted, top_predicted_indices = predicted_probs[0].topk(10, -1)
            # for i in top_predicted_indices:
            #     print(top_texts[int(i.cpu().data.numpy())])

            if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                print(f"beam num = {idx_p}")
            probs_val_debug_loss, _ = predicted_probs.topk(probs.shape[0])
            probs_val_fixed = [round(i.item(), 3) for i in probs_val_debug_loss]
            if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                print(f"text_style_top_{probs.shape[0]}_target_probs = {probs_val_fixed}")

            target = torch.zeros_like(probs[idx_p], device=self.device)
            target[top_indices[idx_p]] = predicted_probs

            target = target.unsqueeze(0)
            cur_text_style_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            text_style_loss += cur_text_style_loss
            losses.append(cur_text_style_loss)
            best_sentences.append(top_texts[torch.argmax(predicted_probs)])

            # debug
            probs_val, indices = predicted_probs.topk(DEBUG_NUM_WORDS)
            debug_best_probs_vals_style.extend(list(probs_val.cpu().data.numpy()))
            style_top_text = [top_texts[i] for i in indices.cpu().data.numpy()]
            debug_best_top_texts_style.extend(style_top_text)

        total_best_sentences_style = {}
        for i in np.argsort(debug_best_probs_vals_style)[-DEBUG_NUM_WORDS:]:
            total_best_sentences_style[debug_best_top_texts_style[i]] = debug_best_probs_vals_style[i]

        loss_string = ''
        for idx_p in range(probs.shape[0]):  # go over all beams
            if idx_p == 0:
                loss_string = f'{losses[0]}'
            else:
                loss_string = loss_string + '%, ' + f'{losses[idx_p]}'

        return text_style_loss, losses, best_sentences, total_best_sentences_style

    def get_text_style_loss(self, probs, context_tokens):
        #use representative vector for calculating the distance between candidates and the representative vecotr
        top_size = TOP_SZIE
        top_probs_LM, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        text_style_loss = 0
        losses = []
        best_sentences = []
        debug_best_top_texts_style = []
        debug_best_probs_vals_style = []
        if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
            print("in text_style loss:")
        style_probs = {}
        for idx_p in range(probs.shape[0]):  # go over all beams
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]:  # go over all optional topk next word
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))

            with torch.no_grad():
                ## based on bert
                #debug
                # inputs = self.text_style_tokenizer(top_texts, padding='max_length', max_length=40, truncation=True, return_tensors="pt")

                #debug

                inputs = self.text_style_tokenizer(top_texts, padding=True, return_tensors="pt")
                inputs['input_ids'] = inputs['input_ids'].to(self.device)
                inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

                if self.model_based_on == 'bert':
                    logits = self.text_style_model(inputs['input_ids'], inputs['attention_mask'])
                elif self.model_based_on == 'clip':
                    logits = self.text_style_model(top_texts)

                # ## based on clip
                # logits = self.text_style_model(top_texts)

                #calculate the distance between the embedding of the text we want to mimic and the all candidated embedding
                #todo:check how to do broadcast with embedding_of_text_to_imitate
                logits.to(self.device)
                self.desired_style_embedding_vector = torch.tensor(self.desired_style_embedding_vector).to(self.device) #todo: check about median instead of mean
                # distances = -abs(logits - self.desired_style_embedding_vector)


                #calc euclidean distance
                distances_from_mean_vec = torch.cdist(self.desired_style_embedding_vector.unsqueeze(0),logits)
                # euclidean_distance = torch.sqrt(torch.sum((logits - self.desired_style_embedding_vector) ** 2, dim=-1))

                #try to calc the distance from  avg+std
                distances = torch.maximum(distances_from_mean_vec -torch.tensor(self.desired_style_embedding_std_vector).to(self.device),torch.tensor(0))


                text_style_grades = nn.functional.softmax(-distances, dim=-1)


                predicted_probs = nn.functional.softmax(text_style_grades / self.text_style_loss_temperature, dim=-1).detach()
                predicted_probs = predicted_probs.type(torch.float32).to(self.device)

                # #####todo: check my sentences
                # my_texts = ["I love you", "I hate you", "It is disgust"]
                # my_texts = ["love", "hate", "disgust", "angry", "cute"]
                # # my_texts = ["you seem like the kind of person i'd like to be best friends with", "it *feels* unethical?"]
                # inputs = self.text_style_tokenizer(my_texts, padding=True, return_tensors="pt")
                # inputs['input_ids'] = inputs['input_ids'].to(self.device)
                # inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
                # my_logits = self.text_style_model(inputs['input_ids'], inputs['attention_mask'])
                # # ## based on clip
                # # logits = self.text_style_model(top_texts)
                #
                # # calculate the distance between the embedding of the text we want to mimic and the all candidated embedding
                # # todo:check how to do broadcast with embedding_of_text_to_imitate
                # logits.to(self.device)
                # self.desired_style_embedding_vector = torch.tensor(self.desired_style_embedding_vector).to(
                #     self.device)  # todo: check about median instead of mean
                # my_distances = -abs(my_logits - self.desired_style_embedding_vector)
                #
                # text_style_grades = nn.functional.softmax(my_distances, dim=-1)[:, 0]
                # text_style_grades = text_style_grades.unsqueeze(0)
                #
                # predicted_probs = nn.functional.softmax(text_style_grades / (self.clip_loss_temperature/50), dim=-1).detach()
                # predicted_probs = predicted_probs.type(torch.float32).to(self.device)
                # pass
                # #####
            #todo: debug
            # val_top_predicted, top_predicted_indices = predicted_probs[0].topk(10, -1)
            # for i in top_predicted_indices:
            #     print(top_texts[int(i.cpu().data.numpy())])

            if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                print(f"beam num = {idx_p}")
            probs_val_debug_loss, _ = predicted_probs[0].topk(probs.shape[0])
            probs_val_fixed = [round(i.item(), 3) for i in probs_val_debug_loss]
            if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                print(f"text_style_top_{probs.shape[0]}_target_probs = {probs_val_fixed}")

            target = torch.zeros_like(probs[idx_p], device=self.device)
            target[top_indices[idx_p]] = predicted_probs[0]

            target = target.unsqueeze(0)
            cur_text_style_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            text_style_loss += cur_text_style_loss
            losses.append(cur_text_style_loss)
            best_sentences.append(top_texts[torch.argmax(predicted_probs[0])])

            # debug
            probs_val, indices = predicted_probs[0].topk(DEBUG_NUM_WORDS)
            debug_best_probs_vals_style.extend(list(probs_val.cpu().data.numpy()))
            style_top_text = [top_texts[i] for i in indices.cpu().data.numpy()]
            debug_best_top_texts_style.extend(style_top_text)

            if self.config.get('update_ViT',False):
                style_probs[idx_p] = target

        total_best_sentences_style = {}
        for i in np.argsort(debug_best_probs_vals_style)[-DEBUG_NUM_WORDS:]:
            total_best_sentences_style[debug_best_top_texts_style[i]] = debug_best_probs_vals_style[i]

        loss_string = ''
        for idx_p in range(probs.shape[0]):  # go over all beams
            if idx_p == 0:
                loss_string = f'{losses[0]}'
            else:
                loss_string = loss_string + '%, ' + f'{losses[idx_p]}'

        return text_style_loss, losses, best_sentences, total_best_sentences_style, style_probs

    def get_text_style_loss_emoji(self, probs, context_tokens):
        top_size = TOP_SZIE
        top_probs_LM, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        text_style_loss = 0
        losses = []
        best_sentences = []
        debug_best_top_texts_style = []
        debug_best_probs_vals_style = []
        for idx_p in range(probs.shape[0]):  # go over all beams
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            # pdb.set_trace()
            for x in top_indices[idx_p]:  # go over all optional topk next word
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))

            with torch.no_grad():
                # top_texts = ["bad day", "It is so sad", "It is disgusting",  "happy day", "wonderful action", "good boy"]
                # tmp_top_texts = ["bad day", "It is so sad", "It is disgusting",  "happy day", "wonderful action", "good boy"]
                # for i in range(len(top_texts)):
                #     top_texts[i] = tmp_top_texts[math.floor(i/100)]
                tokenized, _, _ = self.emoji_st_tokenizer.tokenize_sentences(top_texts)
                tokenized = torch.from_numpy(tokenized.astype(np.int32))

                # tokenized = torch.from_numpy(tokenized.astype(np.int32)).to(self.device)
                # self.emoji_style_model.to(torch.device("cuda"))
                # self.emoji_style_model = self.emoji_style_model.to(self.device)

                # print(f"next(self.emoji_style_model.parameters()).is_cuda = {next(self.emoji_style_model.parameters()).is_cuda}")
                # print(f"tokenized.is_cuda={tokenized.is_cuda}")
                emoji_style_probs = torch.tensor(self.emoji_style_model(tokenized))
                emoji_style_grades = emoji_style_probs[:,self.config['idx_emoji_style_dict'][self.style]].sum(-1)
                emoji_style_grades_normalized = emoji_style_grades/torch.sum(emoji_style_grades)
                # if self.config['use_single_emoji_style']:
                #     desired_labels_idxs = []
                #     for label in self.config['desired_labels']:
                #         desired_labels_idxs.append(self.config['idx_emoji_style_dict'][label])
                #     source_emoji_style_probs = emoji_style_probs[:,desired_labels_idxs]#/0.00000001 #todo:
                #     ###########
                #     source_emoji_style_probs_norm = nn.functional.softmax(source_emoji_style_probs/ EPSILON, dim=-1).detach()+EPSILON
                #     # emoji_style_probs=source_emoji_style_probs_norm
                #     # normalize each row sample
                #     emoji_style_probs = source_emoji_style_probs_norm / torch.unsqueeze(torch.sum(source_emoji_style_probs_norm, dim=-1), 1)
                #     ###########
                #     #normalize each row sample
                #     # emoji_style_probs = emoji_style_probs / torch.unsqueeze(torch.sum(emoji_style_probs,dim=-1), 1)


                # probs = torch.tensor(probs*1000).to(self.device)
                # self.desired_style_embedding_vector = self.desired_style_embedding_vector.to(self.device)

                # emoji_style_loss = ((emoji_style_probs * emoji_style_probs.log()) - (emoji_style_probs * self.desired_style_embedding_vector.log())).sum(-1)
                # predicted_probs = -emoji_style_loss
                # predicted_probs = nn.functional.softmax(predicted_probs, dim=-1).detach()

                # #######
                # #calc euclidean distance
                # distances_from_mean_vec = torch.cdist(self.desired_style_embedding_vector.unsqueeze(0),logits)
                # # euclidean_distance = torch.sqrt(torch.sum((logits - self.desired_style_embedding_vector) ** 2, dim=-1))
                #
                # #try to calc the distance from  avg+std
                # distances = torch.maximum(distances_from_mean_vec -torch.tensor(self.desired_style_embedding_std_vector).to(self.device),torch.tensor(0))
                # ######

                # text_style_grades = nn.functional.softmax(-distances, dim=-1)


                # predicted_probs = nn.functional.softmax(text_style_grades / self.text_style_loss_temperature, dim=-1).detach()
                # predicted_probs = predicted_probs.type(torch.float32).to(self.device)

            ############ debug daniela
            target = torch.zeros_like(probs[idx_p], device=self.device)
            target[top_indices[idx_p]] = emoji_style_grades_normalized.to(self.device)

            target = target.unsqueeze(0)
            cur_text_style_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            text_style_loss += cur_text_style_loss
            losses.append(cur_text_style_loss)
            best_sentences.append(top_texts[torch.argmax(emoji_style_grades_normalized)])

            # debug
            probs_val, indices = emoji_style_grades_normalized.topk(DEBUG_NUM_WORDS)
            debug_best_probs_vals_style.extend(list(probs_val.cpu().data.numpy()))
            style_top_text = [top_texts[i] for i in indices.cpu().data.numpy()]
            debug_best_top_texts_style.extend(style_top_text)

        total_best_sentences_style = {}
        for i in np.argsort(debug_best_probs_vals_style)[-DEBUG_NUM_WORDS:]:
            total_best_sentences_style[debug_best_top_texts_style[i]] = debug_best_probs_vals_style[i]

        loss_string = ''
        for idx_p in range(probs.shape[0]):  # go over all beams
            if idx_p == 0:
                loss_string = f'{losses[0]}'
            else:
                loss_string = loss_string + '%, ' + f'{losses[idx_p]}'
        return text_style_loss, losses, best_sentences, total_best_sentences_style

        # # ############  debug daniela
        #     predicted_probs = predicted_probs.to(self.device)
        #     # target = torch.zeros_like(probs[idx_p], device=self.device)
        #     # target[top_indices[idx_p]] = predicted_probs[0]
        #     # target[top_indices[idx_p]] = predicted_probs
        #
        #     # target = target.unsqueeze(0)
        #     # cur_text_style_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))
        #     cur_text_style_loss = torch.sum(predicted_probs)
        #
        #     text_style_loss += cur_text_style_loss
        #     losses.append(cur_text_style_loss)
        #
        #     # print(f" predicted_probs={ predicted_probs}")
        #     # print(f" len(predicted_probs)={ len(predicted_probs)}")
        #     # print(f" predicted_probs={ predicted_probs}")
        #     # pdb.set_trace()
        #     # best_sentences.append(top_texts[torch.argmax(predicted_probs)])
        #     # best_sentences.append(top_texts[torch.argmax(predicted_probs[0])])
        #
        #     # debug
        #     # probs_val, indices = predicted_probs[0].topk(DEBUG_NUM_WORDS)
        #     # debug_best_probs_vals_style.extend(list(probs_val.cpu().data.numpy()))
        #     # style_top_text = [top_texts[i] for i in indices.cpu().data.numpy()]
        #     # debug_best_top_texts_style.extend(style_top_text)
        #
        # # total_best_sentences_style = {}
        # # for i in np.argsort(debug_best_probs_vals_style)[-DEBUG_NUM_WORDS:]:
        # #     total_best_sentences_style[debug_best_top_texts_style[i]] = debug_best_probs_vals_style[i]
        #
        # loss_string = ''
        # for idx_p in range(probs.shape[0]):  # go over all beams
        #     if idx_p == 0:
        #         loss_string = f'{losses[0]}'
        #     else:
        #         loss_string = loss_string + '%, ' + f'{losses[idx_p]}'
        #
        # # return text_style_loss, losses, best_sentences, total_best_sentences_style
        # return text_style_loss, losses, None, None


    
    def shift_context(self, word_loc, context, last_token, context_tokens, probs_before_shift):
        # contex=past_ke_values of GPT2. tuple of 24, each one composed of 2, s.t. each one of size (5,16,<num of words in context>,64)
        print(f"img_idx={self.img_idx},img_name={self.img_name}, style={self.style}")
        print(f"self.ce_scale,self.clip_scale,self.text_style_scale,self.num_iterations = {self.ce_scale,self.clip_scale,self.text_style_scale,self.num_iterations}")
        print(f"word_loc = {word_loc}")
        context_delta = [tuple([np.zeros(x.shape).astype("float32") for x in p]) for p in context]
        ###get img features
        # self.image_features = self.get_img_feature([self.img_path], None, use_flash_attention=True, return_k_v=False)

        window_mask = torch.ones_like(context[0][0]).to(self.device)

        clip_loss_fixed = 100
        ce_loss_fixed = 100
        text_style_loss_fixed = 100
        i=-1

        th_clip_loss = self.config.get('th_clip_loss',-1)
        th_ce_loss = self.config.get('th_ce_loss',-1)
        th_style_loss = self.config.get('th_style_loss',-1)
        new_weighted_loss = self.config.get('new_weighted_loss',-1)
        max_num_iterations = self.config.get('max_num_iterations',-1)
        last_clip_loss = 1e6
        last_text_style_loss = 1e6
        #todo: check if nee the next line
        self.clip_img = self.clip_preprocess(Image.open(self.img_path)).unsqueeze(0).to(self.device)
        if self.config.get('calc_grad_according_to_first_beam',False):
            probs_before_shift = torch.unsqueeze(probs_before_shift[0][:max_prob_len], 0)  # todo:remove it 9.6.23

        #plot probbilities
        # x = np.arange(0, probs_before_shift.shape[1], 1)  # top_indices[idx_p]
        # for idx_p_i in range(probs_before_shift.shape[0]):
        #     y = probs_before_shift[idx_p_i].cpu().numpy()
        #     plt.figure()
        #     plt.plot(x, y)
        #     plt.title(f"source LM probs for beam_idx={idx_p_i}")
        #     plt.show(block=False)

        while(1):
            i += 1
            # print(f"iteration num: {i}")
            if self.config['print_for_debug']:
                print(f"************** word_loc =  {word_loc}, iter num = {i} **************")
            if new_weighted_loss:
                if clip_loss_fixed<=th_clip_loss and ce_loss_fixed<=th_ce_loss and text_style_loss_fixed<=th_style_loss:
                    break
            else:# not new_weighted_loss:
                pass
            if self.config.get('iterate_until_good_fluency', False):
                if word_loc>=self.config.get('start_word_loc_heavy_iteration', 1):
                    if i == self.config['heavy_max_num_iterations']:
                        break
                elif i == self.num_iterations:
                    break
            elif i == self.num_iterations:
                break
        # for i in range(self.num_iterations):
            # print(f"iter_num =  {i}")
            # if i == self.num_iterations-1:
            #     self.config['print_for_debug'] = source_print_for_debug
            # else:
            #     self.config['print_for_debug'] = False
            self.debug_tracking[word_loc][i] = {}

            curr_shift = [tuple([torch.from_numpy(x).requires_grad_(True).to(device=self.device) for x in p_]) for p_ in
                          context_delta]
            for p0, p1 in curr_shift:
                p0.retain_grad()
                p1.retain_grad()
            shifted_context = list(map(add_context, context, curr_shift))  # array like addition for tuples


            shifted_outputs = self.lm_model(last_token, past_key_values=shifted_context)
            logits = shifted_outputs["logits"][:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            if self.config.get('calc_grad_according_to_first_beam', False):
                probs = torch.  unsqueeze(probs[0][:max_prob_len], 0)  # todo:remove it # 9.6.23

            # # print probs graphs
            if self.config.get('plot_prob_graphs',False):
                # if i>=1:
                if i==4: #plot only last iteration
                    for i_beam in range(probs.shape[0]):
                        # if i_beam>0: #plot only first
                        #     break
                        x = np.arange(0,probs.shape[1],1)#top_indices[idx_p]
                        # Create a grid of subplots
                        fig, axs = plt.subplots(3, 2)

                        # Plot the graphs in separate subplots
                        axs[0, 0].plot(x, probs_before_shift[i_beam].cpu().numpy(), label='source_LM_probs')
                        axs[0, 0].set_title('Source LM Probs')

                        axs[0, 1].plot(x, probs[i_beam].detach().cpu().numpy(), label='fixed_LM_probs')
                        axs[0, 1].set_title('Fixed LM Probs')
                        # clip_target_probs_before_style
                        # if len(sentiment_grades_before_temp.shape)<2:
                        #     sentiment_grades_before_temp = torch.unsqueeze(sentiment_grades_before_temp,1)
                        if len(sentiment_grades_before_temp.shape)>1:
                            sentiment_grades_before_temp=torch.squeeze(sentiment_grades_before_temp)
                        axs[1, 0].plot(x, sentiment_grades_before_temp.cpu().numpy(), label='sentiment_grades_before_temp')
                        axs[1, 0].set_title('sentiment grades before temp')

                        if len(sentiment_grades_after_temp.shape) > 1:
                            sentiment_grades_after_temp = torch.squeeze(sentiment_grades_after_temp)
                        axs[1, 1].plot(x, sentiment_grades_after_temp.cpu().numpy(), label='sentiment_grades_after_temp')
                        axs[1, 1].set_title('sentiment grades after temp')

                        # if len(clip_target_probs_before_style.shape)<2:
                        #     clip_target_probs_before_style = torch.unsqueeze(clip_target_probs_before_style,1)
                        if len(clip_target_probs_before_style.shape) > 1:
                            clip_target_probs_before_style = torch.squeeze(clip_target_probs_before_style)
                        axs[2, 0].plot(x, clip_target_probs_before_style.cpu().detach().numpy(), label='clip_target_probs_before_style')
                        axs[2, 0].set_title('clip target probs before style')

                        axs[2, 1].plot(x, target_probs_clip.cpu().detach().numpy(), label='target_probs_clip')
                        axs[2, 1].set_title('Target Probs Clip')

                        # Add a global title
                        fig.suptitle(f'word loc = {word_loc}, i_beam={i_beam}, iteration number={i}')

                    # Adjust the spacing between subplots
                    plt.tight_layout()

                    plt.show(block=False)



            # x = np.arange(0,probs.shape[1],1)#top_indices[idx_p]
            # for idx_p_i in range(probs.shape[0]):
            #     y = probs[idx_p_i].detach().cpu().numpy()
            #     plt.figure()
            #     plt.plot(x,y)
            #     plt.title(f"fixed LM probs for beam_idx={idx_p_i}")
            #     plt.show(block=False)

            ###################################################
            if word_loc>=self.config.get('start_word_loc_heavy_iteration', 1) and i >= 1:
                if self.config.get('iterate_until_good_fluency', False):
                    with torch.no_grad():
                        top_size = TOP_SZIE
                        top_probs_LM, top_indices = probs.topk(top_size, -1)
                        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in
                                        context_tokens]
                        #########
                        top_texts_for_all_beams = []
                        for idx_p in range(probs.shape[0]):  # for beam search
                            top_texts = []
                            prefix_text = prefix_texts[idx_p]
                            for x in top_indices[idx_p]:
                                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))
                            top_texts_for_all_beams.extend(top_texts)
                        text_features = self.get_txt_features(top_texts_for_all_beams)
                        clip_score_for_all_beams = (self.image_features @ text_features.T)
                        results = self.perplexity.compute(data=top_texts_for_all_beams, model_id='gpt2',
                                                          add_start_token=False)
                        fluency_scores_for_all_beams = torch.tensor(
                            [1 - np.min([res_i, MAX_PERPLEXITY]) / MAX_PERPLEXITY for res_i in
                             results['perplexities']]).to(self.device)
                        style_scores_for_all_beams = self.evaluation_obj['style_classification'].compute_label_for_list(top_texts_for_all_beams, self.style)

                        clip_error = []
                        good_clip_idxs = []
                        for i2 in range(len(clip_score_for_all_beams[0])):
                            if clip_score_for_all_beams[0][i2] >= self.config['desired_min_clip_score']:
                                good_clip_idxs.append(i2)
                            else:
                                # clip_error += self.config['desired_min_clip_score']-top_clip_scores[i]
                                clip_error.append(1 - clip_score_for_all_beams[0][i2] / self.config['desired_min_clip_score'])
                        if len(clip_error) > 0:
                            mean_clip_error = torch.sum(torch.tensor(clip_error))/len(clip_score_for_all_beams[0])
                        else:
                            mean_clip_error = 0

                        fluency_error = []
                        good_fluency_idxs = []
                        for i2 in range(len(fluency_scores_for_all_beams)):
                            if fluency_scores_for_all_beams[i2] > self.config['desired_min_fluency_score']:
                                good_fluency_idxs.append(i2)
                            else:
                                # fluency_error += self.config['desired_min_fluency_score'] - top_fluency_scores[i]
                                fluency_error.append(
                                    1 - fluency_scores_for_all_beams[i2] / self.config['desired_min_fluency_score'])
                        if len(fluency_error) > 0:
                            mean_fluency_error = torch.sum(torch.tensor(fluency_error))/len(fluency_scores_for_all_beams)
                        else:
                            mean_fluency_error = 0

                        style_error = []
                        good_style_idxs = []
                        for i2 in range(len(style_scores_for_all_beams)):
                            if style_scores_for_all_beams[i2] >= self.config['desired_min_style_score']:
                                good_style_idxs.append(i2)
                            else:
                                # style_error += self.config['desired_min_style_score'] - top_style_scores[i]
                                style_error.append(1 - style_scores_for_all_beams[i2] / self.config['desired_min_style_score'])
                        if len(style_error) > 0:
                            mean_style_error = torch.sum(torch.tensor(style_error))/len(style_scores_for_all_beams)
                        else:
                            mean_style_error = 0
                        #find best idxs
                        set_good_idxs = set(good_style_idxs)
                        for l in [good_fluency_idxs, good_clip_idxs]:
                            set_good_idxs = set(set_good_idxs).intersection(l)
                        list_good_idxs = list(set_good_idxs)
                        if len(list_good_idxs)>= self.beam_size:
                            break
                        total_error = mean_clip_error + mean_fluency_error + mean_style_error
                        if total_error > 0:
                            self.clip_scale = float(mean_clip_error / total_error)+EPSILON
                            self.ce_scale = float(mean_fluency_error / total_error)+EPSILON
                            self.text_style_scale = float(mean_style_error / total_error)+EPSILON
                        else:
                            break
                #########
            ###################################################

            loss = 0.0

            #optimize image embedding according to the style
            if self.config.get('update_ViT',False) and word_loc>=self.config['start_loop_clip_style_in_word_num']:#todo check
                # contex=past_ke_values of GPT2. tuple of 24, each one composed of 2, s.t. each one of size (5,16,<num of words in context>,64)
                # update clip
                # self.clip_img = self.clip_preprocess(Image.open(self.img_path)).unsqueeze(0).to(self.device)
                ########## try with continue update CLIP and not rest it every global iteration of ZeroCap
                if i==0: #in the first iteration of generating new token
                    image_fts, k_clip, v_clip = self.clip.encode_image(self.clip_img, return_k_v=True,
                                                                       kv_only_first_layer=self.config[
                                                                           'kv_only_first_layer'])
                    window_mask_clip = torch.ones_like(k_clip[0]).to(self.device)
                    if type(image_fts) == tuple:
                        image_fts = image_fts[0]
                    image_features = sum(image_fts)
                    self.image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    k_delta_clip = [np.zeros(p.shape).astype("float32") for p in k_clip]#context]
                    v_delta_clip = [np.zeros(p.shape).astype("float32") for p in v_clip]#context]

                    curr_shift_k = [torch.from_numpy(p_).requires_grad_(True).to(device=self.device) for p_ in
                                    k_delta_clip]
                    curr_shift_v = [torch.from_numpy(p_).requires_grad_(True).to(device=self.device) for p_ in
                                    v_delta_clip]

                    for (p_k, p_v) in zip(curr_shift_k, curr_shift_v):
                        p_k.retain_grad()
                        p_v.retain_grad()

                    shifted_k_clip = [k_clip[i1] + curr_shift_k[i1] for i1 in range(len(curr_shift_k))]
                    shifted_v_clip = [v_clip[i1] + curr_shift_v[i1] for i1 in range(len(curr_shift_v))]
                    image_fts, k_clip, v_clip = self.clip.encode_image(self.clip_img, updated_k_in=shifted_k_clip,
                                                                       updated_v_in=shifted_v_clip, return_k_v=True,kv_only_first_layer=self.config['kv_only_first_layer'])
                    if type(image_fts) == tuple:
                        image_fts = image_fts[0]
                    image_features = sum(image_fts)
                    self.image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                for clip_update_iter in range(self.config['num_iterations_clip_style']):  # todo: change it self.k=list of 12, each with size of(1,12,50,64)
                    # CLIP LOSS
                    if self.config.get('only_clip_styled_clip_loss'):
                        clip_loss, losses, clip_probs, clip_target_probs_before_style, sentiment_grades_before_temp, sentiment_grades_after_temp = self.get_clip_probs(
                            probs, context_tokens)
                        clip_ViT_loss = clip_loss

                    else:
                        clip_ViT_loss = 0
                        if self.clip_scale != 0:
                            clip_loss, clip_losses, best_sentences_clip, best_sentences_LM, total_best_sentences_clip, total_best_sentences_LM, clip_probs,clip_target_probs_before_style,sentiment_grades_before_temp,sentiment_grades_after_temp = self.clip_loss(
                                probs, context_tokens,grad_lm=False)
                            # if not new_weighted_loss:
                            #     loss += self.clip_scale * clip_loss # change to variable scale
                        # TEXT_STYLE loss:
                        text_style_loss = -100
                        if self.use_style_model and not self.use_text_style_cutting:
                            if self.text_style_scale!=0:
                                total_best_sentences_style = None
                                if self.style_type == 'erc':
                                    text_style_loss, text_style_losses, best_sentences_style, total_best_sentences_style = self.get_text_style_loss_erc(
                                        probs, context_tokens)
                                elif self.style_type == 'clip': #using clip model for text style
                                    text_style_loss, text_style_losses = self.get_text_style_loss_with_clip(probs, context_tokens)
                                elif self.style_type == 'emoji':
                                    text_style_loss, text_style_losses, best_sentences_style, total_best_sentences_style = self.get_text_style_loss_emoji(probs, context_tokens)
                                elif self.style_type == 'style_embed': #my text style embedding that I trained
                                    text_style_loss, text_style_losses, best_sentences_style, total_best_sentences_style, style_probs = self.get_text_style_loss(probs, context_tokens)
                                elif self.style_type == 'roberta':
                                    text_style_loss, text_style_losses, style_probs = self.get_sentiment_loss(probs, context_tokens, self.style)
                                else:
                                    print('check what is the style model!')
                                    exit(-1)

                                #print(f'text_style_loss = {text_style_loss}, text_style_loss_with_scale = {self.text_style_scale * text_style_loss}')
                                # loss += self.text_style_scale * text_style_loss

                        # calc loss according to style
                        clip_style_loss = 0
                        if self.config.get('mul_clip_style_prob',False):
                            print("multiply clip and style probs  in order to influence image embedding to be in specific style")
                        else: #calc CE between clip prob and style prob
                            for idx_p in clip_probs.keys():
                                clip_style_loss += torch.sum(-((style_probs[idx_p]+EPSILON) * torch.log((clip_probs[idx_p]+EPSILON)))) #todo: check if need detach on style tensor
                        # calc loss according to init clip
                        mse_loss = nn.MSELoss()
                        clip_src_clip_loss = mse_loss(self.src_image_features[0], self.image_features)

                        clip_ViT_loss = self.config['loss_scale_style_clip']*clip_style_loss + self.config['loss_scale_src_clip_clip']*clip_src_clip_loss
                        #clip_ViT_loss.retain_grad() #todo: check if need it

                    clip_ViT_loss.backward()

                    ###add the shift to context_clip
                    factor = 1 #1 #todo check
                    # print(f"factor={factor}, global_iteration={i}, update_clip_iter={clip_update_iter}, clip_ViT_loss={clip_ViT_loss}, clip_src_clip_loss={clip_src_clip_loss}, clip_style_loss={clip_style_loss}"
                    print(f"factor={factor}, global_iteration={i}, update_clip_iter={clip_update_iter}, clip_ViT_loss={clip_ViT_loss}, sentiment_temperature={self.config['sentiment_temperature']}")
                    if self.config.get('kv_only_first_layer', False):
                        print(f"num_iterations_clip_style={self.config['num_iterations_clip_style']}, loss_scale_style_clip={self.config['loss_scale_style_clip']}, loss_scale_src_clip_clip={self.config['loss_scale_src_clip_clip']}")

                    # --------- Specific Gen ---------

                    sep_grads_k = None
                    sep_grads_v = None
                    for b in range(k_clip[0].shape[0]): #todo:check it
                        tmp_sep_norms_k = [(torch.norm(x.grad[b:(b + 1)] * window_mask_clip[b:(b + 1)]) + 1e-15) for x in curr_shift_k] #for p_ in curr_shift]

                        tmp_sep_norms_v = [(torch.norm(x.grad[b:(b + 1)] * window_mask_clip[b:(b + 1)]) + 1e-15) for x in curr_shift_v] #for p_ in curr_shift]

                        # normalize gradients
                        tmp_grad_k = [-self.stepsize * factor * (
                                x.grad[b:(b + 1)] * window_mask_clip[b:(b + 1)] / tmp_sep_norms_k[
                            j] ** self.grad_norm_factor).data.cpu().numpy()
                                           for j, x in enumerate(curr_shift_k)] #for i, p_ in enumerate(curr_shift)]
                        tmp_grad_v = [-self.stepsize * factor * (
                                x.grad[b:(b + 1)] * window_mask_clip[b:(b + 1)] / tmp_sep_norms_v[
                            j] ** self.grad_norm_factor).data.cpu().numpy()
                                      for j, x in enumerate(curr_shift_v)]  # for i, p_ in enumerate(curr_shift)]

                        if sep_grads_k is None:
                            sep_grads_k = tmp_grad_k
                        else:
                            for l_index in range(len(sep_grads_k)):
                                sep_grads_k[l_index] = list(sep_grads_k[l_index])
                                for k_index in range(len(sep_grads_k[0])):
                                    sep_grads_k[l_index][k_index] = np.concatenate(
                                        (sep_grads_k[l_index][k_index], tmp_grad_k[l_index][k_index]), axis=0)
                                sep_grads_k[l_index] = tuple(sep_grads_k[l_index])


                        if sep_grads_v is None:
                                sep_grads_v = tmp_grad_v
                        else:
                            for l_index in range(len(sep_grads_v)):
                                sep_grads_v[l_index] = list(sep_grads_v[l_index])
                                for v_index in range(len(sep_grads_v[0])):
                                    sep_grads_v[l_index][v_index] = np.concatenate(
                                        (sep_grads_v[l_index][v_index], tmp_grad_v[l_index][v_index]), axis=0)
                                sep_grads_v[l_index] = tuple(sep_grads_v[l_index])
                    final_grads_k = sep_grads_k
                    final_grads_v = sep_grads_v

                    # --------- update context ---------
                    k_delta_clip = [final_grads_k[i1] + k_delta_clip[i1] for i1 in range(len(k_delta_clip))]
                    v_delta_clip = [final_grads_v[i1] + v_delta_clip[i1] for i1 in range(len(v_delta_clip))]

                    for (p_k, p_v) in zip(curr_shift_k,curr_shift_v):
                        p_k.grad.data.zero_()
                        p_v.grad.data.zero_()

                    new_k_clip = []
                    new_v_clip = []
                    for p_k, p_v in zip(k_clip,v_clip):
                        new_k_clip.append(p_k.detach())
                        new_v_clip.append(p_v.detach())
                    k_clip = new_k_clip
                    v_clip = new_v_clip
                    ##############
                    #
                    # self.image_features, k, v = self.get_img_feature(self.img_path, None, source_clip=False,
                    #                                                  use_flash_attention=False, k=k, v=v,
                    #                                                  return_k_v=True)

                    #update features for next
                    curr_shift_k = [torch.from_numpy(p_).requires_grad_(True).to(device=self.device) for p_ in
                                    k_delta_clip]
                    curr_shift_v = [torch.from_numpy(p_).requires_grad_(True).to(device=self.device) for p_ in
                                    v_delta_clip]

                    for (p_k, p_v) in zip(curr_shift_k, curr_shift_v):
                        p_k.retain_grad()
                        p_v.retain_grad()

                    shifted_k_clip = [k_clip[i1] + curr_shift_k[i1] for i1 in range(len(curr_shift_k))]
                    shifted_v_clip = [v_clip[i1] + curr_shift_v[i1] for i1 in range(len(curr_shift_v))]
                    image_fts, k_clip, v_clip = self.clip.encode_image(self.clip_img, updated_k_in=shifted_k_clip,
                                                                       updated_v_in=shifted_v_clip, return_k_v=True, kv_only_first_layer=self.config['kv_only_first_layer'])

                    if type(image_fts) == tuple:
                        image_fts = image_fts[0]
                    image_features = sum(image_fts)
                    self.image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                print("****************************************")

            if not self.config.get('update_ViT', False):
                # TEXT_STYLE loss:
                text_style_loss = -100
                if self.use_style_model and not self.use_text_style_cutting:
                    if self.text_style_scale != 0:
                        total_best_sentences_style = None
                        if self.style_type == 'erc':
                            text_style_loss, text_style_losses, best_sentences_style, total_best_sentences_style = self.get_text_style_loss_erc(
                                probs, context_tokens)
                        elif self.style_type == 'clip':  # using clip model for text style
                            text_style_loss, text_style_losses = self.get_text_style_loss_with_clip(probs,
                                                                                                    context_tokens)
                        elif self.style_type == 'emoji':
                            text_style_loss, text_style_losses, best_sentences_style, total_best_sentences_style = self.get_text_style_loss_emoji(
                                probs, context_tokens)
                        elif self.style_type == 'style_embed':  # my text style embedding that I trained
                            text_style_loss, text_style_losses, best_sentences_style, total_best_sentences_style, style_probs = self.get_text_style_loss(
                                probs, context_tokens)
                        elif self.style_type == 'roberta':
                            text_style_loss, text_style_losses, style_probs, target_probs_style = self.get_sentiment_loss(probs,
                                                                                                      context_tokens,
                                                                                                      self.style)
                        else:
                            print('check what is the style model!')
                            exit(-1)

                        # print(f'text_style_loss = {text_style_loss}, text_style_loss_with_scale = {self.text_style_scale * text_style_loss}')
                        loss += self.text_style_scale * text_style_loss
                    else:
                        target_probs_style = torch.zeros_like(probs[-1])

            # CLIP LOSS
            if self.clip_scale!=0:
                clip_loss, clip_losses, best_sentences_clip, best_sentences_LM, total_best_sentences_clip,  total_best_sentences_LM, clip_probs, target_probs_clip,clip_target_probs_before_style,sentiment_grades_before_temp ,sentiment_grades_after_temp = self.clip_loss(probs, context_tokens, grad_lm=True)
                # todo: check that clip_probs have grads
                if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                    print("after calc clip loss:")
                clip_loss_fixed = round(clip_loss.item(),3)
                if self.config['print_for_debug']:
                    print(f"{i}: clip_loss = {clip_loss_fixed}")
                clip_losses_fixed = [round(i.item(),3) for i in clip_losses]
                if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                    print(f"clip_losses = {clip_losses_fixed}")
                clip_loss_scale_fixed = round(self.clip_scale * clip_loss.item(),3)
                if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                    print(f"clip_loss with scale = {clip_loss_scale_fixed}")

                if not self.config.get('loss_1_mul_clip_style_in_lm', False):
                    if not new_weighted_loss:
                        loss += self.clip_scale * clip_loss # change to variable scale


                if i == 0: #first iteration
                    LM_0_probs = list(total_best_sentences_LM.values())
                    LM_0_vals = list(total_best_sentences_LM.keys())
                    self.debug_tracking[word_loc][i]['LM_0 - prob'] = LM_0_probs
                    self.debug_tracking[word_loc][i]['LM_0 - val'] = LM_0_vals
                    self.debug_tracking[word_loc][i]['LM - prob'] = list(total_best_sentences_LM.values())
                    self.debug_tracking[word_loc][i]['LM - val'] = list(total_best_sentences_LM.keys())
                    self.debug_tracking[word_loc][i]['CLIP - prob'] = list(total_best_sentences_clip.values())
                    self.debug_tracking[word_loc][i]['CLIP - val'] = list(total_best_sentences_clip.keys())
            else:
                clip_loss, clip_losses = 0,[torch.tensor(0)]*probs.shape[0]
                target_probs_clip = torch.zeros_like(probs[-1])

            # CE/Fluency loss
            if self.ce_scale!=0:
                if self.config.get('loss_1_mul_clip_style_in_lm', False): #todo: continue to debug it
                    probs_before_shift_attention_clip_style = torch.zeros_like(probs_before_shift)
                    for i_b in range(probs.shape[0]):
                        # clip_probs[i_b] = nn.functional.softmax(clip_probs[i_b]/self.config.get('clip_style_temperature',1))
                        # clip_probs[i_b] = nn.functional.softmax(clip_probs[i_b]/0.0001)
                        # clip_probs[i_b] = clip_probs[i_b]/EPSILON
                        # probs_before_shift_attention_clip_style[i_b] = probs_before_shift[i_b] * clip_probs[i_b] + EPSILON
                        probs_before_shift_attention_clip_style[i_b] = probs_before_shift[i_b]*clip_probs[i_b] + clip_probs[i_b]*10
                        norm_val = probs_before_shift_attention_clip_style[i_b].clone()
                        norm_val = norm_val.detach()
                        probs_before_shift_attention_clip_style[i_b] = probs_before_shift_attention_clip_style[i_b]/norm_val.sum()
                    ce_loss_before_scale = ((probs * probs.log()) - (probs * probs_before_shift_attention_clip_style.log())).sum(-1)
                    ce_loss = ((probs * probs.log()) - (probs * probs_before_shift_attention_clip_style.log())).sum(-1)
                    if not new_weighted_loss:
                        loss = ce_loss.sum()
                else:
                    ce_loss_before_scale = ((probs * probs.log()) - (probs * probs_before_shift.log())).sum(-1)
                    ce_loss = self.ce_scale * ((probs * probs.log()) - (probs * probs_before_shift.log())).sum(-1)
                    if not new_weighted_loss:
                        loss += ce_loss.sum()

                ce_losses = (probs * probs_before_shift.log()).sum(-1)
                if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                    print("in ce loss:")
                for i_ce_loss in range(probs.shape[0]):
                    if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                        print(f"beam num = {i_ce_loss}")
                    probs_val, _ = probs[i_ce_loss].topk(probs.shape[0])
                    probs_val_fixed = [round(i_probs_val.item(), 3) for i_probs_val in probs_val]
                    if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                        print(f"ce_top_{probs.shape[0]}_target_probs = {probs_val_fixed}")

                if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                    print("after calc fluency loss:")
                ce_loss_fixed = round(ce_loss_before_scale.sum().item(), 3)
                if self.config['print_for_debug']:
                    print(f"{i}: ce_loss = {ce_loss_fixed}")
                ce_losses_fixed = [round(i_ce_loss_before_scale.item(), 3) for i_ce_loss_before_scale in ce_loss_before_scale]
                if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                    print(f"ce_losses = {ce_losses_fixed}")
                clip_loss_scale_fixed = round(ce_loss.sum().item(), 3)
                if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                    print(f"ce_loss with scale = {clip_loss_scale_fixed}")

            # torch.autograd.set_detect_anomaly(True)
            loss.backward()

            # ---------- Weights ----------
            combined_scores_k = -(ce_loss)
            combined_scores_c = -(self.clip_scale * torch.stack(clip_losses))

            # minmax
            if combined_scores_k.shape[0] == 1:
                tmp_weights_c = tmp_weights_k = torch.ones(*combined_scores_k.shape).to(self.device)
            else:
                tmp_weights_k = ((combined_scores_k - combined_scores_k.min())) / (
                        combined_scores_k.max() - combined_scores_k.min())
                tmp_weights_c = ((combined_scores_c - combined_scores_c.min())) / (
                        combined_scores_c.max() - combined_scores_c.min())

            tmp_weights = 0.5 * tmp_weights_k + 0.5 * tmp_weights_c
            tmp_weights = tmp_weights.view(tmp_weights.shape[0], 1, 1, 1)

            factor = 1

            # --------- Specific Gen ---------
            sep_grads = None

            for b in range(context_tokens.shape[0]):
                tmp_sep_norms = [[(torch.norm(x.grad[b:(b + 1)] * window_mask[b:(b + 1)]) + 1e-15) for x in p_]  #list of size 24, each contains 2 element probably for k and v
                                 for p_ in curr_shift]

                # normalize gradients
                tmp_grad = [tuple([-self.stepsize * factor * (   #list of size 24, each contains 2 element probably for k and v
                        x.grad[b:(b + 1)] * window_mask[b:(b + 1)] / tmp_sep_norms[i][
                    j] ** self.grad_norm_factor).data.cpu().numpy()
                                   for j, x in enumerate(p_)])
                            for i, p_ in enumerate(curr_shift)]
                if sep_grads is None:
                    sep_grads = tmp_grad
                else:
                    for l_index in range(len(sep_grads)):
                        sep_grads[l_index] = list(sep_grads[l_index]) #tuple of 2->list of 2
                        for k_index in range(len(sep_grads[0])):
                            sep_grads[l_index][k_index] = np.concatenate(
                                (sep_grads[l_index][k_index], tmp_grad[l_index][k_index]), axis=0)
                        sep_grads[l_index] = tuple(sep_grads[l_index])
            final_grads = sep_grads #list of size 24, each contains 2 element probably for k and v

            # --------- update context ---------
            context_delta = list(map(add_context, final_grads, context_delta))

            for p0, p1 in curr_shift:
                p0.grad.data.zero_()
                p1.grad.data.zero_()

            new_context = []
            for p0, p1 in context:
                new_context.append((p0.detach(), p1.detach()))
            context = new_context

            ##todo: daniela add break depend on loss
            #weighted_clip_loss = self.clip_scale * clip_loss
            #ce_loss = ce_loss.sum()
            #weighted_text_style_loss = self.text_style_scale * text_style_loss
            #if weighted_clip_loss<=35 and ce_loss<=1.18 and weighted_text_style_loss<=35.5:
            #    break

        print(f"Finished in {i} iterations.")
        if self.config['print_for_debug']:
            print(f'{word_loc+1}/{self.target_seq_length}: clip_loss = {clip_loss}')
            print(f'{word_loc+1}/{self.target_seq_length}: ce_loss = {ce_loss.sum()}')
        if self.use_style_model and not self.use_text_style_cutting:
            if self.config['print_for_debug']:
                print(f'{word_loc+1}/{self.target_seq_length}: style_loss = {text_style_loss}')

        context_delta = [tuple([torch.from_numpy(x).requires_grad_(True).to(device=self.device) for x in p_])
                         for p_ in context_delta]
        context = list(map(add_context, context, context_delta))

        new_context = []
        for p0, p1 in context:
            new_context.append((p0.detach(), p1.detach()))
        context = new_context

        return context

    def update_special_tokens_logits(self, context_tokens, i, logits):
        for beam_id in range(context_tokens.shape[0]):
            for token_idx in set(context_tokens[beam_id][-4:].tolist()):
                factor = self.repetition_penalty if logits[beam_id, token_idx] > 0 else (1 / self.repetition_penalty)
                logits[beam_id, token_idx] /= factor

            if i >= self.ef_idx:
                factor = self.end_factor if logits[beam_id, self.end_token] > 0 else (1 / self.end_factor)
                logits[beam_id, self.end_token] *= factor
            if i == 0:
                start_factor = 1.6
                factor = start_factor if logits[beam_id, self.end_token] > 0 else (1 / start_factor)
                logits[beam_id, self.end_token] /= factor

            for token_idx in list(self.forbidden_tokens):
                factor = self.forbidden_factor if logits[beam_id, token_idx] > 0 else (1 / self.forbidden_factor)
                logits[beam_id, token_idx] /= factor

        return logits

    def clip_loss(self, probs, context_tokens, grad_lm=True):
        '''

        :param probs:
        :param context_tokens:
        :param grad_lm: weaher to condifer grads in LM
        :return:
        '''
        if not self.config.get('update_ViT',False):
            for p_ in self.clip.transformer.parameters(): #todo: check if it defend on text params.
                if p_.grad is not None:
                    p_.grad.data.zero_()

        top_size = TOP_SZIE #512
        top_probs_LM, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        clip_loss = 0
        losses = []
        best_sentences_clip = []
        best_sentences_LM = []
        debug_best_top_texts_clip = []
        debug_best_probs_vals_clip=[]
        debug_best_top_texts_LM = []
        debug_best_probs_vals_LM=[]
        if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
            print("in clip loss:")

        clip_probs = {} #for all beams
        # my debugging of update CLIP
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~") #todo:
        # print("similarity between corrected image to style image:")
        # if self.style == 'positive':
        #     desired_style_image_name = '50.png'
        #     undesired_style_image_name = '51.png'
        #     undesired_style = 'negative'
        # elif self.style == 'negative':
        #     desired_style_image_name = '51.png'
        #     undesired_style_image_name = '50.png'
        #     undesired_style = 'positive'
        # neutral_style_image_name = '49.png'
        # for style_image_name in [desired_style_image_name,undesired_style_image_name,neutral_style_image_name]:
        #     img_path_rel_img = os.path.join(os.path.expanduser('~'), 'data/stylized_images', style_image_name)
        #     clip_img_rel_img = self.clip_preprocess(Image.open(img_path_rel_img)).unsqueeze(0).to(self.device)
        #     image_fts_rel_img = self.clip.encode_image(clip_img_rel_img, return_k_v=False)
        #     if type(image_fts_rel_img) == tuple:
        #         image_fts_rel_img = image_fts_rel_img[0]
        #     image_features_rel_img = sum(image_fts_rel_img)
        #     image_features_rel_img = image_features_rel_img / image_features_rel_img.norm(dim=-1, keepdim=True)
        #     similiraties_rel_img = (self.image_features @ image_features_rel_img) #todo: check if need here transpode T
        #     if style_image_name==desired_style_image_name:
        #         print(f"image similarity to the desired style ({self.style})= {similiraties_rel_img.item()}")
        #     elif style_image_name==undesired_style_image_name:
        #         print(f"image similarity to the undesired style ({undesired_style})= {similiraties_rel_img.item()}")
        #     elif style_image_name==neutral_style_image_name:
        #         print(f"image similarity to the neutral style ('neutral style')= {similiraties_rel_img.item()}")

        # for style_image_name in [desired_style_image_name,undesired_style_image_name,neutral_style_image_name]:
        #     image_features_rel_img = self.get_combined_feature([self.img_path,os.path.join(os.path.expanduser('~'), 'data/stylized_images', neutral_style_image_name), os.path.join(os.path.expanduser('~'), 'data/stylized_images', style_image_name)], [], self.config['arithmetics_weights'], None)
        #     similiraties_rel_img = (self.image_features @ torch.squeeze(image_features_rel_img)) #todo: check if need here transpode T
        #     if style_image_name==desired_style_image_name:
        #         print(f"image similarity to the desired style ({self.style})= {similiraties_rel_img.item()}")
        #     elif style_image_name==undesired_style_image_name:
        #         print(f"image similarity to the undesired style ({undesired_style})= {similiraties_rel_img.item()}")
        #     elif style_image_name==neutral_style_image_name:
        #         print(f"image similarity to the neutral style ('neutral style')= {similiraties_rel_img.item()}")

        # image_features = self.get_combined_feature([self.imgs_path,os.path.join(os.path.expanduser('~'), 'data/stylized_images', desired_style_image_name), os.path.join(os.path.expanduser('~'), 'data/stylized_images', neutral_style_image_name)], [], self.config['arithmetics_weights'], None)
        #for debug update ViT
        # print("similarity between corrected image embedding to text of style name:")
        # text_features_specific_test = self.get_txt_features([self.style])
        # similiraties_specific_test = (self.image_features @ text_features_specific_test.T)
        # print(f"similiraties_specific_test = {similiraties_specific_test.item()}")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        clip_target_probs_before_style = None; sentiment_grades_after_temp=None;sentiment_grades_before_temp=None
        for idx_p in range(probs.shape[0]): # for beam search
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]:
                text = prefix_text + self.lm_tokenizer.decode(x)
                if len(text)>77:
                    text = '.'
                top_texts.append(text)
            ######todo: daniela debug    effect of update CLIP
            # top_texts = ["The bedroom used child abuse"]+["The bedroom of a sweet baby"]
            # if  update_initial_clip: #todo:remove it
            #     for i in range(len(top_texts)):
            #         if i<=len(top_texts)/2:
            #             top_texts[i] = "The bedroom used child abuse"
            #         else:
            #             top_texts[i] = "The bedroom of a sweet baby"
            #     ######todo: daniela debug    effect of update CLIP
            best_sentences_LM.append(prefix_text + self.lm_tokenizer.decode(probs[idx_p].topk(1).indices[0]))

            # grades according to match to style

            probs_val,indices = top_probs_LM[idx_p].topk(DEBUG_NUM_WORDS)
            debug_best_probs_vals_LM.extend(probs_val)
            LM_top_text = [top_texts[i] for i in indices.cpu().data.numpy()]
            debug_best_top_texts_LM.extend(LM_top_text)
            text_features = self.get_txt_features(top_texts)

            if not self.config.get('update_ViT',False):
                with torch.no_grad():
                    similiraties = (self.image_features @ text_features.T)
                    ##### #todo:debug
                    # top_probs_clip, top_indices_clip = similiraties.topk(10, -1)
                    # top_texts_clip = [top_texts[i] for i in top_indices_clip[0]]
                    # print(f"top_texts_clip = {top_texts_clip}")
                    # torch.topk(similiraties, k=10)
                    # style_sco
                    # res[top_indices]

                    #10.5.23
                    if self.config.get('cut_cand2clip',False):
                        #########compute style score for text:
                        # get style score for text
                        with torch.no_grad():
                            results = self.perplexity.compute(data=top_texts, model_id='gpt2',
                                                         add_start_token=False)
                            fluency_scores = torch.tensor([1 - np.min([res_i, MAX_PERPLEXITY]) / MAX_PERPLEXITY for res_i in results['perplexities']]).to(self.device)
                            # self.config['desired_min_fluency_score']
                            # fixed_perplexity = 1 - np.min([results['perplexities'][0], MAX_PERPLEXITY]) / MAX_PERPLEXITY

                            # inputs = self.sentiment_tokenizer(top_texts, padding=True, return_tensors="pt")
                            if self.config['style_type'] == 'roberta':
                                inputs = self.sentiment_tokenizer(text_list, padding=True,
                                                                         return_tensors='pt').to(
                                    self.device)
                            # elif self.config['style_type'] == 'emoji':
                            #     inputs = self.emoji_st_tokenizer.tokenize_sentences(top_texts).to(self.device)
                            inputs['input_ids'] = inputs['input_ids'].to(self.sentiment_model.device)
                            inputs['attention_mask'] = inputs['attention_mask'].to(self.sentiment_model.device)
                            logits = self.sentiment_model(**inputs)['logits']
                            sentiment_grades = None
                            ########
                            positive_grades = nn.functional.softmax(logits, dim=-1)[:, 2]
                            neutral_grades = nn.functional.softmax(logits, dim=-1)[:, 1]
                            negative_grades = nn.functional.softmax(logits, dim=-1)[:, 0]
                            style_scores = []
                            if self.style == 'positive':
                                desired_scores = positive_grades
                                undesired_scores = negative_grades
                            elif self.style == 'negative':
                                desired_scores = negative_grades
                                undesired_scores = positive_grades
                            for i in range(len(desired_scores)):
                                # if desired_scores[i] > 2*undesired_scores[i] and desired_scores[i]> 2* neutral_grades[i]:
                                if desired_scores[i] > 0.5 and fluency_scores[i] > self.config['desired_min_fluency_score']:
                                    style_scores.append(1)
                                else:
                                    style_scores.append(0)

                            # if self.style == 'positive':
                            #     sentiment_grades = nn.functional.softmax(logits, dim=-1)[:, 2]
                            # elif self.style == 'neutral':
                            #     sentiment_grades = nn.functional.softmax(logits, dim=-1)[:, 1]
                            # elif self.style == 'negative':
                            #     sentiment_grades = nn.functional.softmax(logits, dim=-1)[:, 0]
                        #########
                        # style_scores = [1 for i in outputs_bin if i == self.desired_style_bin]

                        # style_scores = torch.tensor([1 if i>0.5  else 0 for i in sentiment_grades]).to(self.device)
                        style_scores = torch.tensor(style_scores).to(self.device)
                        if self.device == "cuda":
                            similiraties = torch.mul(similiraties, style_scores)
                        else:
                            similiraties = np.multiply(similiraties, style_scores)
                    # end - 10.5.23
                    if self.use_text_style_cutting:
                        #if self.check_if_cut_score[idx_p]:
                        if self.check_if_cut_score:
                            cut_scores = True
                            similarity_topk_vals, similarity_topk_indices = similiraties[0].topk(self.config.get('requires_num_min_clip_score_val',-1)[self.style])

                            for i in similarity_topk_vals:
                                if i <= self.config.get('requires_min_clip_score_val',-1)[self.style]:
                                    cut_scores = False
                            if cut_scores:
                                # print("~~~~~")
                                # print(f"similarity_topk_vals={similarity_topk_vals}")
                                # print("~~~~~")
                                self.check_if_cut_score = False
                        if not self.check_if_cut_score:
                            # print("~~~~~")
                            if self.config['style_type'] == 'emoji':
                                ############ top_texts[0] = "In love"; top_texts[1] = "In hate"
                                tokenized, _, _ = self.emoji_st_tokenizer.tokenize_sentences(top_texts)
                                tokenized = torch.from_numpy(tokenized.astype(np.int32))
                                emoji_style_probs = torch.tensor(self.emoji_style_model(tokenized))
                                emoji_style_grades = emoji_style_probs[:,
                                                     self.config['idx_emoji_style_dict'][self.style]].sum(-1)
                                if self.config.get('style_mul_not_cut',False):
                                    style_scores = (emoji_style_grades / torch.sum(emoji_style_grades)).to(self.device)
                                else:
                                    emoji_style_grades_cutted = [0] * len(emoji_style_grades)
                                    for i in range(len(emoji_style_grades)):
                                        if emoji_style_grades[i] > self.config['threshold_sentiment'][self.style] and similiraties[0][i]>=self.config.get('requires_min_clip_score_val',-1)[self.style]:  # todo
                                            # if emoji_style_grades[i]>0.3:
                                            # print(f"i={i},emoji_style_grades[i]={emoji_style_grades[i]},top_texts[i]={top_texts[i]}")
                                            emoji_style_grades_cutted[i] = 1
                                    style_scores = torch.tensor(emoji_style_grades_cutted).to(self.device)
                                ############
                            elif self.config['style_type'] == 'style_embed':
                                outputs_bin = self.text_style_cls_model.compute_label_for_list(top_texts)
                                style_scores = [1 for i in outputs_bin if i == self.desired_style_bin]

                            # top_probs_style, top_indices_style = style_scores.sort(descending=True)
                            # top_texts_emoji_style = [top_texts[i] for i in top_indices_style]
                            # clip_prob_by_top_style_cls = [similiraties[0][i].item() for i in top_indices_style]

                            # good_style_idxs = (style_scores == 1).nonzero(as_tuple=True)[0]
                            # top_texts_emoji_style = [top_texts[i] for i in good_style_idxs]
                            # print(f"top_texts_emoji_style = {top_texts_emoji_style}")
                            # similiraties[good_style_idxs]
                            #####
                            # similiraties = similiraties * style_scores
                            #zero sentences which are not in the desired style

                            if self.device == "cuda":
                                similiraties = torch.mul(similiraties, style_scores)
                            else:
                                similiraties = np.multiply(similiraties, style_scores)
                        # top_probs_indices_clip_emoji_style, top_indices_clip_emoji_style = similiraties.topk(10, -1)
                        # top_texts_clip_emoji_style = [top_texts[i] for i in top_indices_clip_emoji_style[0]]
                        # print(f"top_texts_clip_emoji_style = {top_texts_clip_emoji_style}")
                    ############
                    # top_texts = ['Ugly and disgusting  image', 'Beautiful and amazing image']
                    # top_texts = ['The wonderful line waiting in the baggage carousel.',
                    #              'A suitcase devastated the platform at Penn Station in New York City.']
                    # pos_text = ['positive']
                    if self.text_style_list:
                        text_style_features = self.get_txt_features([self.text_style_list])
                        similarity_to_style = text_style_features @ text_features.T
                        similarity_to_style_normalized = similarity_to_style / similarity_to_style[0].norm(dim=-1)
                        #add affect with the style:
                        image_text_similiraties = (self.image_features @ text_features.T)
                        if self.device=="cuda":
                            similiraties = torch.mul(image_text_similiraties, similarity_to_style_normalized)
                        else:
                            similiraties = np.multiply(image_text_similiraties, similarity_to_style_normalized)
                        ######

                    target_probs = nn.functional.softmax(similiraties / self.clip_loss_temperature, dim=-1).detach()
                    target_probs = target_probs.type(torch.float32)

                    if self.config.get('mul_clip_style',False):
                        ########adding style effect#todo
                        text_list = self.preprocess_text_for_roberta(top_texts)
                        if self.config['style_type'] == 'roberta':
                            encoded_input = self.sentiment_tokenizer(text_list, padding=True, return_tensors='pt').to(
                                self.device)
                            output = self.sentiment_model(**encoded_input)
                            scores = output[0].detach()
                            scores = nn.functional.softmax(scores, dim=-1) #get grades for each image
                            sentiment_grades_before_temp = nn.functional.softmax(scores, dim=0)
                            scores = nn.functional.softmax(scores/self.sentiment_temperature, dim=0) #rank grades between all images
                            # sentiment_grades = None
                            if self.style == 'positive':
                                sentiment_grades = scores[:, 2]
                                sentiment_grades_before_temp=sentiment_grades_before_temp[:,2]
                            elif self.style == 'neutral':
                                sentiment_grades = scores[:, 1]
                                sentiment_grades_before_temp = sentiment_grades_before_temp[:, 2]
                            elif self.style == 'negative':
                                sentiment_grades = scores[:, 0]
                                sentiment_grades_before_temp = sentiment_grades_before_temp[:, 2]
                            sentiment_grades = sentiment_grades.unsqueeze(0)
                            sentiment_grades_before_temp = sentiment_grades_before_temp.unsqueeze(0)
                        elif self.config['style_type'] == 'emoji':
                            tokenized, _, _ = self.emoji_st_tokenizer.tokenize_sentences(top_texts)
                            tokenized = torch.from_numpy(tokenized.astype(np.int32))
                            emoji_style_probs = torch.tensor(self.emoji_style_model(tokenized)).to(self.device)
                            scores = emoji_style_probs[:, self.config['idx_emoji_style_dict'][self.style]].sum(-1)
                            # emoji_style_grades_normalized = emoji_style_grades / torch.sum(emoji_style_grades)
                            sentiment_grades_before_temp = nn.functional.softmax(scores, dim=0)
                            scores = nn.functional.softmax(scores / self.sentiment_temperature,
                                                           dim=0)  # rank grades between all images
                            sentiment_grades = scores.unsqueeze(0).to(self.device)

                        clip_target_probs = target_probs
                        clip_target_probs_before_style = clip_target_probs
                        sentiment_grades_after_temp = sentiment_grades
                        clip_target_probs_weightes_style = sentiment_grades * clip_target_probs
                        clip_target_probs_weightes_style_normalized = clip_target_probs_weightes_style/clip_target_probs_weightes_style.sum()

                        target_probs = clip_target_probs_weightes_style_normalized
            else: #'update_ViT'=True: collect grad
                similiraties = (self.image_features @ text_features.T)
                ##### #todo:debug
                # top_probs_clip, top_indices_clip = similiraties.topk(10, -1)
                # top_texts_clip = [top_texts[i] for i in top_indices_clip[0]]
                # print(f"top_texts_clip = {top_texts_clip}")
                # torch.topk(similiraties, k=10)
                # style_sco
                # res[top_indices]

                # 10.5.23
                if self.config.get('cut_cand2clip',False):
                    #########compute style score for text:
                    # get style score for text
                    with torch.no_grad():
                        results = self.perplexity.compute(data=top_texts, model_id='gpt2',
                                                          add_start_token=False)
                        fluency_scores = torch.tensor([1 - np.min([res_i, MAX_PERPLEXITY]) / MAX_PERPLEXITY for res_i in
                                                       results['perplexities']]).to(self.device)
                        # self.config['desired_min_fluency_score']
                        # fixed_perplexity = 1 - np.min([results['perplexities'][0], MAX_PERPLEXITY]) / MAX_PERPLEXITY
                        if self.config['style_type'] == 'roberta':
                            encoded_input = self.sentiment_tokenizer(text_list, padding=True, return_tensors='pt').to(
                                self.device)
                        # elif self.config['style_type'] == 'emoji':
                        #     encoded_input = self.emoji_st_tokenizer.tokenize_sentences(top_texts).to(self.device)
                        inputs['input_ids'] = inputs['input_ids'].to(self.sentiment_model.device)
                        inputs['attention_mask'] = inputs['attention_mask'].to(self.sentiment_model.device)
                        logits = self.sentiment_model(**inputs)['logits']
                        sentiment_grades = None
                        ########
                        positive_grades = nn.functional.softmax(logits, dim=-1)[:, 2]
                        neutral_grades = nn.functional.softmax(logits, dim=-1)[:, 1]
                        negative_grades = nn.functional.softmax(logits, dim=-1)[:, 0]
                        style_scores = []
                        if self.style == 'positive':
                            desired_scores = positive_grades
                            undesired_scores = negative_grades
                        elif self.style == 'negative':
                            desired_scores = negative_grades
                            undesired_scores = positive_grades
                        for i in range(len(desired_scores)):
                            # if desired_scores[i] > 2*undesired_scores[i] and desired_scores[i]> 2* neutral_grades[i]:
                            if desired_scores[i] > 0.5 and fluency_scores[i] > self.config['desired_min_fluency_score']:
                                style_scores.append(1)
                            else:
                                style_scores.append(0)

                        # if self.style == 'positive':
                        #     sentiment_grades = nn.functional.softmax(logits, dim=-1)[:, 2]
                        # elif self.style == 'neutral':
                        #     sentiment_grades = nn.functional.softmax(logits, dim=-1)[:, 1]
                        # elif self.style == 'negative':
                        #     sentiment_grades = nn.functional.softmax(logits, dim=-1)[:, 0]
                    #########
                    # style_scores = [1 for i in outputs_bin if i == self.desired_style_bin]

                    # style_scores = torch.tensor([1 if i>0.5  else 0 for i in sentiment_grades]).to(self.device)
                    style_scores = torch.tensor(style_scores).to(self.device)
                    if self.device == "cuda":
                        similiraties = torch.mul(similiraties, style_scores)
                    else:
                        similiraties = np.multiply(similiraties, style_scores)
                # end - 10.5.23
                if self.use_text_style_cutting:
                    # if self.check_if_cut_score[idx_p]:
                    if self.check_if_cut_score:
                        cut_scores = True
                        similarity_topk_vals, similarity_topk_indices = similiraties[0].topk(
                            self.config.get('requires_num_min_clip_score_val',-1)[self.style])

                        for i in similarity_topk_vals:
                            if i <= self.config.get('requires_min_clip_score_val',-1)[self.style]:
                                cut_scores = False
                        if cut_scores:
                            # print("~~~~~")
                            # print(f"similarity_topk_vals={similarity_topk_vals}")
                            # print("~~~~~")
                            self.check_if_cut_score = False
                    if not self.check_if_cut_score:
                        # print("~~~~~")
                        if self.config['style_type'] == 'emoji':
                            ############ top_texts[0] = "In love"; top_texts[1] = "In hate"
                            tokenized, _, _ = self.emoji_st_tokenizer.tokenize_sentences(top_texts)
                            tokenized = torch.from_numpy(tokenized.astype(np.int32))
                            emoji_style_probs = torch.tensor(self.emoji_style_model(tokenized))
                            emoji_style_grades = emoji_style_probs[:,
                                                 self.config['idx_emoji_style_dict'][self.style]].sum(-1)
                            if self.config.get('style_mul_not_cut',False):
                                style_scores = (emoji_style_grades / torch.sum(emoji_style_grades)).to(self.device)
                            else:
                                emoji_style_grades_cutted = [0] * len(emoji_style_grades)
                                for i in range(len(emoji_style_grades)):
                                    if emoji_style_grades[i] > self.config['threshold_sentiment'][self.style] and \
                                            similiraties[0][i] >= self.config.get('requires_min_clip_score_val',-1)[
                                        self.style]:  # todo
                                        # if emoji_style_grades[i]>0.3:
                                        # print(f"i={i},emoji_style_grades[i]={emoji_style_grades[i]},top_texts[i]={top_texts[i]}")
                                        emoji_style_grades_cutted[i] = 1
                                style_scores = torch.tensor(emoji_style_grades_cutted).to(self.device)
                            ############
                        elif self.config['style_type'] == 'style_embed':
                            outputs_bin = self.text_style_cls_model.compute_label_for_list(top_texts)
                            style_scores = [1 for i in outputs_bin if i == self.desired_style_bin]

                        # top_probs_style, top_indices_style = style_scores.sort(descending=True)
                        # top_texts_emoji_style = [top_texts[i] for i in top_indices_style]
                        # clip_prob_by_top_style_cls = [similiraties[0][i].item() for i in top_indices_style]

                        # good_style_idxs = (style_scores == 1).nonzero(as_tuple=True)[0]
                        # top_texts_emoji_style = [top_texts[i] for i in good_style_idxs]
                        # print(f"top_texts_emoji_style = {top_texts_emoji_style}")
                        # similiraties[good_style_idxs]
                        #####
                        # similiraties = similiraties * style_scores
                        # zero sentences which are not in the desired style

                        if self.device == "cuda":
                            similiraties = torch.mul(similiraties, style_scores)
                        else:
                            similiraties = np.multiply(similiraties, style_scores)
                    # top_probs_indices_clip_emoji_style, top_indices_clip_emoji_style = similiraties.topk(10, -1)
                    # top_texts_clip_emoji_style = [top_texts[i] for i in top_indices_clip_emoji_style[0]]
                    # print(f"top_texts_clip_emoji_style = {top_texts_clip_emoji_style}")
                if self.text_style_list:
                    text_style_features = self.get_txt_features([self.text_style_list])
                    similarity_to_style = text_style_features @ text_features.T
                    similarity_to_style_normalized = similarity_to_style / similarity_to_style[0].norm(dim=-1)
                    # add affect with the style:
                    image_text_similiraties = (self.image_features @ text_features.T)
                    if self.device == "cuda":
                        similiraties = torch.mul(image_text_similiraties, similarity_to_style_normalized)
                    else:
                        similiraties = np.multiply(image_text_similiraties, similarity_to_style_normalized)
                    ######


                similiraties = similiraties.float() #todo: check if need / self.clip_loss_temperature
                target_probs = nn.functional.softmax(similiraties / self.clip_loss_temperature, dim=-1) # .detach() todo: check if it is ok
                target_probs = target_probs.type(torch.float32)

                if self.config.get('mul_clip_style', False):
                    ########adding style effect#todo
                    with torch.no_grad():
                        text_list = self.preprocess_text_for_roberta(top_texts)
                        if self.config['style_type'] == 'roberta':
                            encoded_input = self.sentiment_tokenizer(text_list, padding=True, return_tensors='pt').to(
                                self.device)
                            output = self.sentiment_model(**encoded_input)
                            scores = output[0].detach()
                            scores = nn.functional.softmax(scores, dim=-1)  # get grades for each image
                            sentiment_grades_before_temp = nn.functional.softmax(scores, dim=0)
                            scores = nn.functional.softmax(scores / self.sentiment_temperature,
                                                           dim=0)  # rank grades between all images
                            # sentiment_grades = None
                            if self.style == 'positive':
                                sentiment_grades = scores[:, 2]
                                sentiment_grades_before_temp = sentiment_grades_before_temp[:, 2]
                            elif self.style == 'neutral':
                                sentiment_grades = scores[:, 1]
                                sentiment_grades_before_temp = sentiment_grades_before_temp[:, 2]
                            elif self.style == 'negative':
                                sentiment_grades = scores[:, 0]
                                sentiment_grades_before_temp = sentiment_grades_before_temp[:, 2]
                            sentiment_grades = sentiment_grades.unsqueeze(0)
                            sentiment_grades_before_temp = sentiment_grades_before_temp.unsqueeze(0)
                        elif self.config['style_type'] == 'emoji':
                            tokenized, _, _ = self.emoji_st_tokenizer.tokenize_sentences(top_texts)
                            tokenized = torch.from_numpy(tokenized.astype(np.int32))
                            emoji_style_probs = torch.tensor(self.emoji_style_model(tokenized)).to(self.device)
                            scores = emoji_style_probs[:, self.config['idx_emoji_style_dict'][self.style]].sum(-1)
                            # emoji_style_grades_normalized = emoji_style_grades / torch.sum(emoji_style_grades)
                            sentiment_grades_before_temp = nn.functional.softmax(scores, dim=0)
                            scores = nn.functional.softmax(scores / self.sentiment_temperature,
                                                           dim=0)  # rank grades between all images
                            sentiment_grades = scores.unsqueeze(0).to(self.device)

                    clip_target_probs = target_probs
                    clip_target_probs_before_style = clip_target_probs
                    sentiment_grades_after_temp = sentiment_grades
                    clip_target_probs_weightes_style = sentiment_grades * clip_target_probs
                    clip_target_probs_weightes_style_normalized = clip_target_probs_weightes_style / clip_target_probs_weightes_style.sum()

                    target_probs = clip_target_probs_weightes_style_normalized

            if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                print(f"beam num = {idx_p}")
            # target_probs = target_probs[0]
            probs_val_debug_loss, _ = target_probs.topk(probs.shape[0])
            try:
                probs_val_fixed = [round(i.item(),3) for i in probs_val_debug_loss]
            except:
                probs_val_fixed = [round(i.item(), 3) for i in probs_val_debug_loss[0]]
            if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
                print(f"clip_top_{probs.shape[0]}_target_probs = {probs_val_fixed}")

            target = torch.zeros_like(probs[idx_p])
            target[top_indices[idx_p]] = target_probs
            target = target.unsqueeze(0)

            sentiment_grades_before_temp_t = torch.zeros_like(probs[idx_p])
            if sentiment_grades_before_temp is not None:
                sentiment_grades_before_temp_t[top_indices[idx_p]] = sentiment_grades_before_temp
            sentiment_grades_before_temp_t = sentiment_grades_before_temp_t.unsqueeze(0)

            sentiment_grades_after_temp_t = torch.zeros_like(probs[idx_p])
            if sentiment_grades_after_temp is not None:
                sentiment_grades_after_temp_t[top_indices[idx_p]] = sentiment_grades_after_temp
            sentiment_grades_after_temp_t = sentiment_grades_after_temp_t.unsqueeze(0)

            clip_target_probs_before_style_t = torch.zeros_like(probs[idx_p])
            if clip_target_probs_before_style is not None:
                clip_target_probs_before_style_t[top_indices[idx_p]] = clip_target_probs_before_style
            clip_target_probs_before_style_t = clip_target_probs_before_style_t.unsqueeze(0)

            if grad_lm:
                cur_clip_loss = torch.sum(-(target.detach() * torch.log(probs[idx_p:(idx_p + 1)]))) #todo check grad becuase ViT - I added .detach()
            else:
                cur_clip_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)]).detach())) #todo check grad becuase ViT - I added .detach()

            clip_loss += cur_clip_loss
            losses.append(cur_clip_loss)

            # x = np.arange(0, probs.shape[1], 1)  # top_indices[idx_p]
            # y = target[0].cpu().numpy()
            # plt.figure()
            # plt.plot(x, y)
            # plt.title(f"CLIP probs for beam_idx={idx_p}")
            # plt.show(block=False)
            #debug
            try:
                probs_val, indices = target_probs.topk(DEBUG_NUM_WORDS)
                best_sentences_clip.append(top_texts[torch.argmax(target_probs)])
            except:
                probs_val, indices = target_probs[0].topk(DEBUG_NUM_WORDS)
                best_sentences_clip.append(top_texts[torch.argmax(target_probs[0])])

            # if len(indices[0])>1:
            #     indices = indices[0]
            # debug_best_probs_vals_clip.extend(list(probs_val.cpu().data.numpy()))
            # clip_top_text = [top_texts[i] for i in indices.cpu().data.numpy()]
            # debug_best_top_texts_clip.extend(clip_top_text)

            if self.config.get('update_ViT',False) or self.config.get('loss_1_mul_clip_style_in_lm',False) :
                clip_probs[idx_p] = target

        debug_best_probs_vals_LM = [float(i.cpu().data.numpy()) for i in debug_best_probs_vals_LM]

        total_best_sentences_clip = {}
        total_best_sentences_LM = {}
        # for i in np.argsort(debug_best_probs_vals_clip)[-DEBUG_NUM_WORDS:]:
        #     total_best_sentences_clip[debug_best_top_texts_clip[i]] = debug_best_probs_vals_clip[i]
        # for i in np.argsort(debug_best_probs_vals_LM)[-DEBUG_NUM_WORDS:]:
        #     total_best_sentences_LM[debug_best_top_texts_LM[i]] = debug_best_probs_vals_LM[i]
        return clip_loss, losses, best_sentences_clip, best_sentences_LM, total_best_sentences_clip, total_best_sentences_LM, clip_probs, target[0], clip_target_probs_before_style_t,sentiment_grades_before_temp_t,sentiment_grades_after_temp_t


    def get_clip_probs(self, probs, context_tokens):
        if not self.config.get('update_ViT',False):
            for p_ in self.clip.transformer.parameters(): #todo: check if it defend on text params.
                if p_.grad is not None:
                    p_.grad.data.zero_()
        top_size = TOP_SZIE #512
        top_probs_LM, top_indices = probs.topk(top_size, -1)
        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]
        clip_loss = 0
        losses = []
        if self.config['print_for_debug'] and self.config['print_for_debug_redundant']:
            print("in clip loss:")

        clip_probs = {} #for all beams
        clip_target_probs_before_style = None; sentiment_grades_after_temp=None;sentiment_grades_before_temp=None
        for idx_p in range(probs.shape[0]): # for beam search
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]:
                text = prefix_text + self.lm_tokenizer.decode(x)
                if len(text)>77:
                    text = ''
                top_texts.append(text)

            # grades according to match to style
            probs_val,indices = top_probs_LM[idx_p].topk(DEBUG_NUM_WORDS)
            text_features = self.get_txt_features(top_texts)

            #similarities according to fixed clip
            similiraties = (self.image_features @ text_features.T)
            similiraties = similiraties.float() #todo: check if need / self.clip_loss_temperature
            target_probs_fixed = nn.functional.softmax(similiraties / self.clip_loss_temperature, dim=-1) # .detach() todo: check if it is ok
            # target_probs_fixed = torch.unsqueeze(target_probs_fixed,1)
            target_probs_fixed = target_probs_fixed.type(torch.float32)

            #get vector probs of source clip multiply with style
            with torch.no_grad():
                similiraties_source = (self.src_image_features.detach() @ text_features.T.detach()) #todo:check if detach ~~~~~~~~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                similiraties_source = similiraties_source.float()  # todo: check if need / self.clip_loss_temperature
                target_probs_source = nn.functional.softmax(similiraties_source / self.clip_loss_temperature,
                                                     dim=-1)  # .detach() todo: check if it is ok
                target_probs_source = target_probs_source.type(torch.float32)
                ########adding style effect#todo
                text_list = self.preprocess_text_for_roberta(top_texts)
                if self.config['style_type'] == 'roberta':
                    encoded_input = self.sentiment_tokenizer(text_list, padding=True, return_tensors='pt').to(
                        self.device)
                    output = self.sentiment_model(**encoded_input)
                    scores = output[0].detach()
                    scores = nn.functional.softmax(scores, dim=-1)  # get grades for each image
                    sentiment_grades_before_temp = nn.functional.softmax(scores, dim=0)
                    scores = nn.functional.softmax(scores / self.sentiment_temperature,
                                                   dim=0)  # rank grades between all images
                    # sentiment_grades = None
                    if self.style == 'positive':
                        sentiment_grades = scores[:, 2]
                        sentiment_grades_before_temp = sentiment_grades_before_temp[:, 2]
                    elif self.style == 'neutral':
                        sentiment_grades = scores[:, 1]
                        sentiment_grades_before_temp = sentiment_grades_before_temp[:, 2]
                    elif self.style == 'negative':
                        sentiment_grades = scores[:, 0]
                        sentiment_grades_before_temp = sentiment_grades_before_temp[:, 2]
                    sentiment_grades = sentiment_grades.unsqueeze(0)
                elif self.config['style_type'] == 'emoji':
                    for i_text,text in enumerate(top_texts):
                        if '\t' in text:
                            top_texts[i_text] = text.replace("\t", " " )
                        if '\n' in text:
                            top_texts[i_text] = text.replace("\n", " ")
                        if text=='':
                            top_texts[i_text] = ' '
                    tokenized, _, _ = self.emoji_st_tokenizer.tokenize_sentences(top_texts)
                    tokenized = torch.from_numpy(tokenized.astype(np.int32))
                    emoji_style_probs = torch.tensor(self.emoji_style_model(tokenized)).to(self.device)
                    scores = emoji_style_probs[:, self.config['idx_emoji_style_dict'][self.style]].sum(-1)
                    # emoji_style_grades_normalized = emoji_style_grades / torch.sum(emoji_style_grades)
                    sentiment_grades_before_temp = nn.functional.softmax(scores, dim=0)
                    scores = nn.functional.softmax(scores / self.sentiment_temperature,
                                                   dim=0)  # rank grades between all images
                    sentiment_grades = scores.unsqueeze(0).to(self.device)

            clip_target_probs_source = target_probs_source
            clip_target_probs_before_style = clip_target_probs_source
            sentiment_grades_after_temp = sentiment_grades
            clip_target_probs_weightes_style = sentiment_grades * clip_target_probs_source
            clip_target_probs_weightes_style_normalized = clip_target_probs_weightes_style / clip_target_probs_weightes_style.sum()

            source_target_with_style = torch.zeros_like(probs[idx_p])+EPSILON
            source_target_with_style[top_indices[idx_p]] = clip_target_probs_weightes_style_normalized
            source_target_with_style = source_target_with_style.unsqueeze(0)

            target_fixed = torch.zeros_like(probs[idx_p])+EPSILON
            target_fixed[top_indices[idx_p]] = target_probs_fixed
            target_fixed = target_fixed.unsqueeze(0)

            cur_clip_loss = torch.sum((target_fixed * torch.log(target_fixed.detach()))) + torch.sum(-(target_fixed * torch.log(source_target_with_style.detach())))  # todo check i need the first element
            clip_loss += cur_clip_loss
            losses.append(cur_clip_loss)
            if self.config.get('update_ViT',False) or self.config.get('loss_1_mul_clip_style_in_lm',False) :
                clip_probs[idx_p] = source_target_with_style
        return clip_loss, losses, clip_probs, clip_target_probs_before_style,sentiment_grades_before_temp,sentiment_grades_after_temp
