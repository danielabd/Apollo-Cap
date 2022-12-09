#try to use it: for adapting to python3
#https: // github.com / sks3i / pycocoevalcap


#https://github.com/wangleihitcs/CaptionMetrics
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor
from nltk import word_tokenize
'''
def blue(reference_sentence_list, candidate_sen):
    reference = [sentence.split() for sentence in reference_sentence_list]
    candidate = candidate_sen.split()
    return sentence_bleu(reference, candidate)


def METEOR(reference_sentence_list, candidate_sen):
    reference = [word_tokenize(sentence) for sentence in reference_sentence_list]
    candidate = word_tokenize(candidate_sen)
    return sentence_bleu(reference, candidate)
'''

from zero_shot_style.pycocoevalcap.bleu.bleu import Bleu
from zero_shot_style.pycocoevalcap.cider.cider import Cider
from zero_shot_style.pycocoevalcap.meteor.meteor import Meteor
from zero_shot_style.pycocoevalcap.rouge.rouge import Rouge
from zero_shot_style.pycocoevalcap.spice import Spice
import json
from model.ZeroCLIP import CLIPTextGenerator
import os

def bleu(gts, res):
    print("Calculate bleu score...")
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    print('belu = %s' % score)
    return score

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


def sentence_fluency():
    srilm_path =
    pass


def style_accuracy():
    pass

def diversitiy():
    pass


def main():
    cuda_idx = "1"
    #with open('examples/gts.json', 'r') as file:
        #gts = json.load(file) #dictionary s.t. each key is the image name and value is a list of sentences
    #with open('examples/res.json', 'r') as file:
        #res = json.load(file) #dictionary s.t. each key is the image name and value is a list of the resulted sentence
    img_path = os.path.join(os.path.expanduser('~'), 'zero-shot-style', 'data', 'imgs','47.jpeg')
    gts = {1:["the cat sat on the mat","the cat on the mat"]}
    res = {1:["the cat sat on the mat"]}
    bleu_score = bleu(gts, res)
    print(f"bleu score  = {bleu_score}")
    cider_score = cider(gts, res)
    print(f"cider score  = {cider_score}")
    rouge_score = rouge(gts, res)
    print(f"rouge score  = {rouge_score}")
    #meteor_score = meteor(gts, res)
    #print(f"meteor score  = {meteor_score}")
    #spice_score = spice(gts, res)
    text_generator = CLIPTextGenerator(cuda_idx=cuda_idx)
    CLIPScoreRef_score = CLIPScoreRef(res, gts, text_generator)
    print(f"CLIPScoreRef  = {CLIPScoreRef_score}")
    CLIPScore_score = CLIPScore(text_generator, img_path, res)
    print(f"CLIPScore  = {CLIPScore_score}")

    sentence_fluency()
    style_accuracy()
    diversitiy()# or maybe creativity
    print('Finished to evaluate')

if __name__=='__main__':
    main()