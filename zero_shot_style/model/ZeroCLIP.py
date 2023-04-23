# import pdb
import csv
import math
import os.path
import heapq
# import pdb

import numpy as np
from torch import nn
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt_neo import GPTNeoForCausalLM
import torch
import clip
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
                 clip_checkpoints=os.path.join(os.path.expanduser('~'),'projects/zero-shot-style/zero_shot_style','clip_checkpoints'),
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
                 **kwargs):

        self.style_type = None
        self.config = config
        self.use_text_style_cutting = use_text_style_cutting
        if evaluation_obj:
            evaluation_obj = evaluation_obj
        if config:
            self.model_based_on = config['model_based_on']
        else:
            self.model_based_on = model_based_on
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
                                                    download_root=clip_checkpoints, jit=False)
        # convert_models_to_fp32(self.clip)
        self.clip.eval()

        # Init arguments
        self.target_seq_length = int(target_seq_length)
        self.reset_context_delta = reset_context_delta
        self.num_iterations = int(num_iterations)
        self.clip_loss_temperature = clip_loss_temperature
        self.text_style_loss_temperature = text_style_loss_temperature
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
        task = 'sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        self.sentiment_model_name = MODEL
        self.sentiment_model = ''  # todo: remove it
        if False:  # todo: remove it
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()

            # SENTIMENT: Freeze sentiment model weights
            for param in self.sentiment_model.parameters():
                param.requires_grad = False

            # SENTIMENT: tokenizer for sentiment analysis module
            self.sentiment_tokenizer_name = self.sentiment_model_name
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_tokenizer_name)

            # SENTIMENT: fields for type and scale of sentiment
            self.sentiment_scale = 1
            self.sentiment_type = 'none'

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

    def set_params(self, ce_scale, clip_scale, text_style_scale, beam_size,  num_iterations):
        self.ce_scale = ce_scale
        self.clip_scale = clip_scale
        self.text_style_scale = text_style_scale
        self.beam_size = beam_size
        self.num_iterations = num_iterations

    def get_debug_tracking(self):
        return self.debug_tracking


    def get_img_feature(self, img_path, weights, source_clip = False):
        #imgs = [Image.fromarray(cv2.imread(x)) for x in img_path]
        #imgs = [Image.fromarray(cv2.imread(x).astype('uint8'), 'RGB') for x in img_path]
        #imgs = [Image.fromarray(cv2.imread(x), 'RGB') for x in img_path]
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]

        with torch.no_grad():
            if self.model_based_on == 'bert' or source_clip:
                image_fts = [self.clip.encode_image(x) for x in clip_imgs]
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

    def run(self, image_features, cond_text, beam_size, text_style_scale = None, text_to_imitate = None, desired_style_embedding_vector = None, desired_style_embedding_std_vector = None, style_type = None,img_idx=None, img_name=None, style=None, desired_style_bin=False):
    
        # SENTIMENT: sentiment_type can be one of ['positive','negative','neutral', 'none']
        self.image_features = image_features
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
            self.style_type = style_type #'clip','twitter','emotions' , 'erc
            if self.style_type != 'erc':
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
                                                                            max_length=512, truncation=True,
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
        
    # SENTIMENT: function we added for changing the result to the requested sentiment
    def get_sentiment_loss(self, probs, context_tokens,sentiment_type): 
        top_size = 512
        _, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        sentiment_loss = 0
        losses = []

        for idx_p in range(probs.shape[0]): #go over all beams
          
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]: #go over all optional topk next word
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))
            
            
            #get score for text
            with torch.no_grad():
               
                inputs = self.sentiment_tokenizer(top_texts, padding=True, return_tensors="pt")
                inputs['input_ids'] = inputs['input_ids'].to(self.sentiment_model.device)
                inputs['attention_mask'] = inputs['attention_mask'].to(self.sentiment_model.device)
                logits = self.sentiment_model(**inputs)['logits']
                                   
                sentiment_grades = None
                if sentiment_type=='positive':
                        sentiment_grades= nn.functional.softmax(logits, dim=-1)[:,2]
                elif sentiment_type=='neutral':
                        sentiment_grades= nn.functional.softmax(logits, dim=-1)[:,1]
                elif sentiment_type=='negative':
                        sentiment_grades= nn.functional.softmax(logits, dim=-1)[:,0]
                sentiment_grades = sentiment_grades.unsqueeze(0)
                
                predicted_probs = nn.functional.softmax(sentiment_grades / self.clip_loss_temperature, dim=-1).detach()
                predicted_probs = predicted_probs.type(torch.float32).to(self.device)
             
            
            target = torch.zeros_like(probs[idx_p], device=self.device)
            target[top_indices[idx_p]] = predicted_probs[0]
            
            target = target.unsqueeze(0)
            cur_sentiment_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))
            
            
            sentiment_loss += cur_sentiment_loss
            losses.append(cur_sentiment_loss)
        
        loss_string = ''
        for idx_p in range(probs.shape[0]): #go over all beams
            if idx_p==0:
                loss_string = f'{losses[0]}'
            else:
                loss_string = loss_string+'%, '+f'{losses[idx_p]}'
            
        return sentiment_loss, losses

    def get_text_style_loss_with_clip(self, probs, context_tokens):
        for p_ in self.clip.transformer.parameters():
            if p_.grad is not None:
                p_.grad.data.zero_()

        top_size = 512
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
        top_size = 512
        top_probs_LM, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        text_style_loss = 0
        losses = []
        best_sentences = []
        debug_best_top_texts_style = []
        debug_best_probs_vals_style = []
        if self.config['print_for_debug']:
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

            if self.config['print_for_debug']:
                print(f"beam num = {idx_p}")
            probs_val_debug_loss, _ = predicted_probs.topk(probs.shape[0])
            probs_val_fixed = [round(i.item(), 3) for i in probs_val_debug_loss]
            if self.config['print_for_debug']:
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
        top_size = 512
        top_probs_LM, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        text_style_loss = 0
        losses = []
        best_sentences = []
        debug_best_top_texts_style = []
        debug_best_probs_vals_style = []
        if self.config['print_for_debug']:
            print("in text_style loss:")
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

            if self.config['print_for_debug']:
                print(f"beam num = {idx_p}")
            probs_val_debug_loss, _ = predicted_probs[0].topk(probs.shape[0])
            probs_val_fixed = [round(i.item(), 3) for i in probs_val_debug_loss]
            if self.config['print_for_debug']:
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

    def get_text_style_loss_emoji(self, probs, context_tokens):
        top_size = 512
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
        print(f"img_idx={self.img_idx},img_name={self.img_name}, style={self.style}")
        print(f"self.ce_scale,self.clip_scale,self.text_style_scale,self.num_iterations = {self.ce_scale,self.clip_scale,self.text_style_scale,self.num_iterations}")
        context_delta = [tuple([np.zeros(x.shape).astype("float32") for x in p]) for p in context]

        window_mask = torch.ones_like(context[0][0]).to(self.device)

        for i in range(self.num_iterations):
            self.debug_tracking[word_loc][i] = {}
            curr_shift = [tuple([torch.from_numpy(x).requires_grad_(True).to(device=self.device) for x in p_]) for p_ in
                          context_delta]

            for p0, p1 in curr_shift:
                p0.retain_grad()
                p1.retain_grad()

            shifted_context = list(map(add_context, context, curr_shift))

            shifted_outputs = self.lm_model(last_token, past_key_values=shifted_context)
            logits = shifted_outputs["logits"][:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)

            loss = 0.0

            # CLIP LOSS
            if self.clip_scale!=0:
                clip_loss, clip_losses, best_sentences_clip, best_sentences_LM, total_best_sentences_clip,  total_best_sentences_LM = self.clip_loss(probs, context_tokens)
                if self.config['print_for_debug']:
                    print("after calc clip loss:")
                clip_loss_fixed = round(clip_loss.item(),3)
                if self.config['print_for_debug']:
                    print(f"clip_loss = {clip_loss_fixed}")
                clip_losses_fixed = [round(i.item(),3) for i in clip_losses]
                if self.config['print_for_debug']:
                    print(f"clip_losses = {clip_losses_fixed}")
                clip_loss_scale_fixed = round(self.clip_scale * clip_loss.item(),3)
                if self.config['print_for_debug']:
                    print(f"clip_loss with scale = {clip_loss_scale_fixed}")
                loss += self.clip_scale * clip_loss
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

            # CE/Fluency loss
            if self.ce_scale!=0:
                ce_loss_before_scale = ((probs * probs.log()) - (probs * probs_before_shift.log())).sum(-1)
                ce_loss = self.ce_scale * ((probs * probs.log()) - (probs * probs_before_shift.log())).sum(-1)
                loss += ce_loss.sum()
                ce_losses = (probs * probs_before_shift.log()).sum(-1)
                if self.config['print_for_debug']:
                    print("in ce loss:")
                for i_ce_loss in range(probs.shape[0]):
                    if self.config['print_for_debug']:
                        print(f"beam num = {i_ce_loss}")
                    probs_val, _ = probs[i_ce_loss].topk(probs.shape[0])
                    probs_val_fixed = [round(i_probs_val.item(), 3) for i_probs_val in probs_val]
                    if self.config['print_for_debug']:
                        print(f"ce_top_{probs.shape[0]}_target_probs = {probs_val_fixed}")

                if self.config['print_for_debug']:
                    print("after calc fluency loss:")
                ce_loss_fixed = round(ce_loss_before_scale.sum().item(), 3)
                if self.config['print_for_debug']:
                    print(f"ce_loss = {ce_loss_fixed}")
                ce_losses_fixed = [round(i_ce_loss_before_scale.item(), 3) for i_ce_loss_before_scale in ce_loss_before_scale]
                if self.config['print_for_debug']:
                    print(f"ce_losses = {ce_losses_fixed}")
                clip_loss_scale_fixed = round(ce_loss.sum().item(), 3)
                if self.config['print_for_debug']:
                    print(f"ce_loss with scale = {clip_loss_scale_fixed}")

            # TEXT_STYLE:
            if self.use_style_model and not self.use_text_style_cutting:
                text_style_loss=-100
                if self.text_style_scale!=0:
                    if self.style_type == 'erc':
                        text_style_loss, text_style_losses, best_sentences_style, total_best_sentences_style = self.get_text_style_loss_erc(
                            probs, context_tokens)
                    elif self.style_type == 'clip': #using clip model for text style
                        text_style_loss, text_style_losses = self.get_text_style_loss_with_clip(probs, context_tokens)
                    elif self.style_type == 'emoji':
                        text_style_loss, text_style_losses, best_sentences_style, total_best_sentences_style = self.get_text_style_loss_emoji(probs, context_tokens)
                    else: #my text style embedding that I trained
                        text_style_loss, text_style_losses, best_sentences_style, total_best_sentences_style = self.get_text_style_loss(probs, context_tokens)
                    #print(f'text_style_loss = {text_style_loss}, text_style_loss_with_scale = {self.text_style_scale * text_style_loss}')
                    # loss += self.text_style_scale * text_style_loss
                    loss += self.text_style_scale * text_style_loss
                    if self.config['print_for_debug']:
                        print("after calc text_style loss:")
                    text_style_loss_fixed = round(text_style_loss.item(), 3)
                    if self.config['print_for_debug']:
                        print(f"text_style_loss = {text_style_loss_fixed}")
                    text_style_losses_fixed = [round(i.item(), 3) for i in text_style_losses]
                    if self.config['print_for_debug']:
                        print(f"text_style_losses = {text_style_losses_fixed}")
                    text_style_loss_scaled_fixed = round(self.text_style_scale * text_style_loss.item(), 3)
                    if self.config['print_for_debug']:
                        print(f"text_style_loss with scale = {text_style_loss_scaled_fixed}")


                    if total_best_sentences_style:
                        self.debug_tracking[word_loc][i]['STYLE - prob'] = list(total_best_sentences_style.values())
                        self.debug_tracking[word_loc][i]['STYLE - val'] = list(total_best_sentences_style.keys())

                # tmp_text_loss[iteration_num][beam_num][text / ce_loss / clip_loss / style_loss]
                #for beam_num in range(len(best_sentences_LM)):
                #    self.tmp_text_loss[cur_iter][beam_num]['clip_text'] = best_sentences_clip[beam_num]
                #    self.tmp_text_loss[cur_iter][beam_num]['clip_loss'] = clip_losses[beam_num]
                #    self.tmp_text_loss[cur_iter][beam_num]['style_text'] = best_sentences_style[beam_num]
                #    self.tmp_text_loss[cur_iter][beam_num]['style_loss'] = text_style_losses[beam_num]
                #    self.tmp_text_loss[cur_iter][beam_num]['ce_text'] = best_sentences_LM[beam_num]
                #    self.tmp_text_loss[cur_iter][beam_num]['ce_loss'] = ce_losses[beam_num]
                #write_tmp_text_loss(self.tmp_text_loss)
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
                tmp_sep_norms = [[(torch.norm(x.grad[b:(b + 1)] * window_mask[b:(b + 1)]) + 1e-15) for x in p_]
                                 for p_ in curr_shift]

                # normalize gradients
                tmp_grad = [tuple([-self.stepsize * factor * (
                        x.grad[b:(b + 1)] * window_mask[b:(b + 1)] / tmp_sep_norms[i][
                    j] ** self.grad_norm_factor).data.cpu().numpy()
                                   for j, x in enumerate(p_)])
                            for i, p_ in enumerate(curr_shift)]
                if sep_grads is None:
                    sep_grads = tmp_grad
                else:
                    for l_index in range(len(sep_grads)):
                        sep_grads[l_index] = list(sep_grads[l_index])
                        for k_index in range(len(sep_grads[0])):
                            sep_grads[l_index][k_index] = np.concatenate(
                                (sep_grads[l_index][k_index], tmp_grad[l_index][k_index]), axis=0)
                        sep_grads[l_index] = tuple(sep_grads[l_index])
            final_grads = sep_grads

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

        if self.config['print_for_debug']:
            print(f'{word_loc+1}/{self.target_seq_length}: clip_loss_with_scale = {self.clip_scale * clip_loss}')
            print(f'{word_loc+1}/{self.target_seq_length}: ce_loss = {ce_loss.sum()}')
        if self.use_style_model and not self.use_text_style_cutting:
            if self.config['print_for_debug']:
                print(f'{word_loc+1}/{self.target_seq_length}: style_loss_with_scale = {self.text_style_scale * text_style_loss}')

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

    def clip_loss(self, probs, context_tokens):
        for p_ in self.clip.transformer.parameters():
            if p_.grad is not None:
                p_.grad.data.zero_()

        top_size = 512
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
        if self.config['print_for_debug']:
            print("in clip loss:")
        for idx_p in range(probs.shape[0]): # for beam search
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]:
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))
            best_sentences_LM.append(prefix_text + self.lm_tokenizer.decode(probs[idx_p].topk(1).indices[0]))

            # grades according to match to style

            probs_val,indices = top_probs_LM[idx_p].topk(DEBUG_NUM_WORDS)
            debug_best_probs_vals_LM.extend(probs_val)
            LM_top_text = [top_texts[i] for i in indices.cpu().data.numpy()]
            debug_best_top_texts_LM.extend(LM_top_text)
            text_features = self.get_txt_features(top_texts)

            with torch.no_grad():
                similiraties = (self.image_features @ text_features.T)

                ##### #todo:debug
                # top_probs_clip, top_indices_clip = similiraties.topk(10, -1)
                # top_texts_clip = [top_texts[i] for i in top_indices_clip[0]]
                # print(f"top_texts_clip = {top_texts_clip}")
                # torch.topk(similiraties, k=10)
                # style_sco
                # res[top_indices]

                if self.use_text_style_cutting:
                    #if self.check_if_cut_score[idx_p]:
                    if self.check_if_cut_score:
                        cut_scores = True
                        similarity_topk_vals, similarity_topk_indices = similiraties[0].topk(self.config['requires_num_min_clip_score_val'][self.style])

                        for i in similarity_topk_vals:
                            if i <= self.config['requires_min_clip_score_val'][self.style]:
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
                            if self.config['style_mul_not_cut']:
                                style_scores = (emoji_style_grades / torch.sum(emoji_style_grades)).to(self.device)
                            else:
                                emoji_style_grades_cutted = [0] * len(emoji_style_grades)
                                for i in range(len(emoji_style_grades)):
                                    if emoji_style_grades[i] > self.config['threshold_sentiment'][self.style] and similiraties[0][i]>=self.config['requires_min_clip_score_val'][self.style]:  # todo
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
            if self.config['print_for_debug']:
                print(f"beam num = {idx_p}")
            probs_val_debug_loss, _ = target_probs[0].topk(probs.shape[0])
            probs_val_fixed = [round(i.item(),3) for i in probs_val_debug_loss]
            if self.config['print_for_debug']:
                print(f"clip_top_{probs.shape[0]}_target_probs = {probs_val_fixed}")

            target = torch.zeros_like(probs[idx_p])
            target[top_indices[idx_p]] = target_probs[0]
            target = target.unsqueeze(0)
            cur_clip_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            clip_loss += cur_clip_loss
            losses.append(cur_clip_loss)
            best_sentences_clip.append(top_texts[torch.argmax(target_probs[0])])
            #debug
            probs_val, indices = target_probs[0].topk(DEBUG_NUM_WORDS)
            debug_best_probs_vals_clip.extend(list(probs_val.cpu().data.numpy()))
            clip_top_text = [top_texts[i] for i in indices.cpu().data.numpy()]
            debug_best_top_texts_clip.extend(clip_top_text)

        debug_best_probs_vals_LM = [float(i.cpu().data.numpy()) for i in debug_best_probs_vals_LM]

        total_best_sentences_clip = {}
        total_best_sentences_LM = {}
        for i in np.argsort(debug_best_probs_vals_clip)[-DEBUG_NUM_WORDS:]:
            total_best_sentences_clip[debug_best_top_texts_clip[i]] = debug_best_probs_vals_clip[i]
        for i in np.argsort(debug_best_probs_vals_LM)[-DEBUG_NUM_WORDS:]:
            total_best_sentences_LM[debug_best_top_texts_LM[i]] = debug_best_probs_vals_LM[i]
        return clip_loss, losses, best_sentences_clip, best_sentences_LM, total_best_sentences_clip, total_best_sentences_LM
