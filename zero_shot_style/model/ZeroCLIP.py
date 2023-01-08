import os.path
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
from zero_shot_style.model.TextStyleEmbedding_2_1_2023 import TextStyleEmbed
import pickle

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
                 forbidden_tokens_file_path='./forbidden_tokens.npy',
                 clip_checkpoints='./clip_checkpoints',
                 target_seq_length=15,
                 reset_context_delta=True,
                 num_iterations=5,
                 clip_loss_temperature=0.01,
                 text_style_loss_temperature = 0.0002,
                 clip_scale=1.,
                 ce_scale=0.2,
                 stepsize=0.3,#todo
                 grad_norm_factor=0.9,
                 fusion_factor=0.99,
                 repetition_penalty=1.,
                 end_token='.',
                 end_factor=1.01,
                 forbidden_factor=20,
                 cuda_idx = None,
                 model_path = None,
                 tmp_text_loss = None,
                 use_style_model = False,
                 **kwargs):

        self.tmp_text_loss = tmp_text_loss
        self.device = f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu"#todo: change

        
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
        self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=self.device,
                                                    download_root=clip_checkpoints, jit=False)
        # convert_models_to_fp32(self.clip)

        # Init arguments
        self.target_seq_length = target_seq_length
        self.reset_context_delta = reset_context_delta
        self.num_iterations = num_iterations
        self.clip_loss_temperature = clip_loss_temperature
        self.text_style_loss_temperature = text_style_loss_temperature
        self.clip_scale = clip_scale
        self.ce_scale = ce_scale
        self.stepsize = stepsize
        self.grad_norm_factor = grad_norm_factor
        self.fusion_factor = fusion_factor
        self.repetition_penalty = repetition_penalty
        self.end_token = self.lm_tokenizer.encode(end_token)[0]
        self.end_factor = end_factor
        self.ef_idx = 1
        self.forbidden_factor = forbidden_factor
        
        
        # SENTIMENT: adding the sentiment model 
        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        

        self.sentiment_model_name = MODEL
        self.sentiment_model = '' #todo: remove it
        if False: #todo: remove it
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()

            # SENTIMENT: Freeze sentiment model weights
            for param in self.sentiment_model.parameters():
                param.requires_grad = False
            
            # SENTIMENT: tokenizer for sentiment analysis module
            self.sentiment_tokenizer_name =  self.sentiment_model_name
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_tokenizer_name)

            # SENTIMENT: fields for type and scale of sentiment
            self.sentiment_scale = 1
            self.sentiment_type = 'none'

        # TEXT STYLE: adding the text style model
        self.use_text_style = True
        #self.text_style_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_style_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.text_style_scale = 1
        # MODEL = '/home/bdaniela/zero-shot-style/zero_shot_style/model/data/2_classes_trained_model_emotions.pth'

        self.text_style_model_name = model_path
        #self.text_style_model = AutoModelForSequenceClassification.from_pretrained(self.text_style_model_name)

        self.use_style_model = use_style_model
        if self.use_style_model:
            self.text_style_model = TextStyleEmbed(device=self.device)
            LR = 1e-4
            # optimizer = SGD(self.text_style_model.parameters(), lr=LR) #check if to remove mark
            checkpoint = torch.load(self.text_style_model_name, map_location='cuda:0')
            self.text_style_model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #check if to remove mark

            self.text_style_model.to(self.device)
            self.text_style_model.eval()

            #self.text_style_model = torch.load(self.text_style_model_name)

            #self.text_style_model.to(self.device)
            # self.text_style_model.eval()

            # TEXT_STYLE: Freeze text style model weights
            for param in self.text_style_model.parameters():
                param.requires_grad = False

            self.desired_style_embedding_vector = ''
            # TEXT_STYLE: tokenizer for text style analysis module
            #self.text_style_tokenizer_name = self.text_style_model_name
            #self.text_style_tokenizer = AutoTokenizer.from_pretrained(self.text_style_tokenizer_name)





    def get_img_feature(self, img_path, weights):
        #imgs = [Image.fromarray(cv2.imread(x)) for x in img_path]
        #imgs = [Image.fromarray(cv2.imread(x).astype('uint8'), 'RGB') for x in img_path]
        #imgs = [Image.fromarray(cv2.imread(x), 'RGB') for x in img_path]
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]

        with torch.no_grad():
            image_fts = [self.clip.encode_image(x) for x in clip_imgs]

            if weights is not None:
                image_features = sum([x * weights[i] for i, x in enumerate(image_fts)])
            else:
                image_features = sum(image_fts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.detach()

    def get_txt_features(self, text):
        clip_texts = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            text_features = self.clip.encode_text(clip_texts)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach()

    def get_combined_feature(self, img_path, texts, weights_i, weights_t):
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]
        clip_texts = [clip.tokenize(x).to(self.device) for x in texts]

        with torch.no_grad():
            image_fts = [self.clip.encode_image(x) for x in clip_imgs]
            text_fts = [self.clip.encode_text(x) for x in clip_texts]

            features = sum([x * weights_i[i] for i, x in enumerate(image_fts)])
            if weights_t is not None:
                features += sum([x * weights_t[i] for i, x in enumerate(text_fts)])

            features = features / features.norm(dim=-1, keepdim=True)
            return features.detach()

    def run(self, image_features, cond_text, beam_size, sentiment_type = None, sentiment_scale = None, text_style_scale = None, text_to_imitate = None, desired_style_embedding_vector = None, style_type = None):
    
        # SENTIMENT: sentiment_type can be one of ['positive','negative','neutral', 'none']
        self.image_features = image_features
        if self.use_style_model:
            self.sentiment_type = sentiment_type
            self.sentiment_scale = sentiment_scale
            self.text_style_scale = text_style_scale
            self.style_type = style_type #'clip','twitter','emotions'
            if not text_to_imitate:
                self.desired_style_embedding_vector = desired_style_embedding_vector
            else: #there is text_to_imitate:
                #use clip features
                if style_type=='clip':#'clip','twitter','emotions'
                    self.text_style_features = self.get_txt_features(text_to_imitate)
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

    def get_text_style_loss(self, probs, context_tokens):
        top_size = 512
        _, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        text_style_loss = 0
        losses = []
        best_sentences = []
        for idx_p in range(probs.shape[0]):  # go over all beams
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]:  # go over all optional topk next word
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))

            # get score for text
            with torch.no_grad():
                ## based on bert
                inputs = self.text_style_tokenizer(top_texts, padding=True, return_tensors="pt")
                inputs['input_ids'] = inputs['input_ids'].to(self.device)
                inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
                logits = self.text_style_model(inputs['input_ids'], inputs['attention_mask'])
                # ## based on clip
                # logits = self.text_style_model(top_texts)

                #calculate the distance between the embedding of the text we want to mimic and the all candidated embedding
                #todo:check how to do broadcast with embedding_of_text_to_imitate
                logits.to(self.device)
                self.desired_style_embedding_vector = torch.tensor(self.desired_style_embedding_vector).to(self.device) #todo: check about median instead of mean
                distances = -abs(logits - self.desired_style_embedding_vector)

                text_style_grades = nn.functional.softmax(distances, dim=-1)[:, 0]
                text_style_grades = text_style_grades.unsqueeze(0)

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

            target = torch.zeros_like(probs[idx_p], device=self.device)
            target[top_indices[idx_p]] = predicted_probs[0]

            target = target.unsqueeze(0)
            cur_text_style_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            text_style_loss += cur_text_style_loss
            losses.append(cur_text_style_loss)
            best_sentences.append(top_texts[torch.argmax(predicted_probs[0])])

        loss_string = ''
        for idx_p in range(probs.shape[0]):  # go over all beams
            if idx_p == 0:
                loss_string = f'{losses[0]}'
            else:
                loss_string = loss_string + '%, ' + f'{losses[idx_p]}'

        return text_style_loss, losses, best_sentences


    
    def shift_context(self, word_loc, context, last_token, context_tokens, probs_before_shift):
        context_delta = [tuple([np.zeros(x.shape).astype("float32") for x in p]) for p in context]

        window_mask = torch.ones_like(context[0][0]).to(self.device)

        for i in range(self.num_iterations):
        #cur_iter=-1
        #tmp_text_loss = {}
        #while(1):
        #    cur_iter=cur_iter+1
        #    if cur_iter>5: #todo: change
        #        break
            #print(f' iteration num = {cur_iter}')

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
            clip_loss, clip_losses, best_sentences_clip, best_sentences_LM = self.clip_loss(probs, context_tokens)
            loss += self.clip_scale * clip_loss

            # CE/Fluency loss
            ce_loss = self.ce_scale * ((probs * probs.log()) - (probs * probs_before_shift.log())).sum(-1)
            loss += ce_loss.sum()
            ce_losses = (probs * probs_before_shift.log()).sum(-1)

            if self.use_style_model:
                # SENTIMENT: adding the sentiment component
                if self.sentiment_type!='none':
                    sentiment_loss, sentiment_losses = self.get_sentiment_loss(probs, context_tokens,self.sentiment_type)
                    #print(f'sentiment_loss = {sentiment_loss}, sentiment_loss_with_scale = {self.sentiment_scale * sentiment_loss}')
                    loss += self.sentiment_scale * sentiment_loss

                # TEXT_STYLE: adding the text_style component
                if self.use_text_style:
                    if self.style_type == 'clip': #using clip model for text style
                        text_style_loss, text_style_losses = self.get_text_style_loss_with_clip(probs, context_tokens)
                    else:
                        text_style_loss, text_style_losses, best_sentences_style = self.get_text_style_loss(probs, context_tokens)
                    #print(f'text_style_loss = {text_style_loss}, text_style_loss_with_scale = {self.text_style_scale * text_style_loss}')
                    loss += self.text_style_scale * text_style_loss

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

        print(f'{word_loc}: clip_loss_with_scale = {self.clip_scale * clip_loss}')
        print(f'{word_loc}: ce_loss = {ce_loss.sum()}')
        if self.use_style_model:
            print(f'{word_loc}: style_loss_with_scale = {self.text_style_scale * text_style_loss}')

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
        _, top_indices = probs.topk(top_size, -1)
        
        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        clip_loss = 0
        losses = []
        best_sentences_clip = []
        best_sentences_LM = []
        for idx_p in range(probs.shape[0]): # for beam search
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]:
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))
            best_sentences_LM.append(prefix_text + self.lm_tokenizer.decode(probs[idx_p].topk(1).indices[0]))
            text_features = self.get_txt_features(top_texts)

            with torch.no_grad():
                similiraties = (self.image_features @ text_features.T)
                target_probs = nn.functional.softmax(similiraties / self.clip_loss_temperature, dim=-1).detach()
                target_probs = target_probs.type(torch.float32)
            target = torch.zeros_like(probs[idx_p])
            target[top_indices[idx_p]] = target_probs[0]
            target = target.unsqueeze(0)
            cur_clip_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            clip_loss += cur_clip_loss
            losses.append(cur_clip_loss)
            best_sentences_clip.append(top_texts[torch.argmax(target_probs[0])])
        return clip_loss, losses, best_sentences_clip, best_sentences_LM
