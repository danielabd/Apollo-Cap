import argparse
import torch
import clip
from model.ZeroCLIP import CLIPTextGenerator
from datetime import datetime
import os.path
import csv
from collections import defaultdict
import numpy as np
import pickle
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    # parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--cond_text", type=str, default="")
    parser.add_argument("--reset_context_delta", action="store_true",
                        help="Should we reset the context at each token gen")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=2)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'arithmetics'])

    parser.add_argument("--caption_img_path", type=str, default='example_images/captions/COCO_val2014_000000008775.jpg',
                        help="Path to image for captioning")

    parser.add_argument("--arithmetics_imgs", nargs="+",
                        default=['example_images/arithmetics/woman2.jpg',
                                 'example_images/arithmetics/king2.jpg',
                                 'example_images/arithmetics/man2.jpg'])
    parser.add_argument("--arithmetics_weights", nargs="+", default=[1, 1, -1])

    args = parser.parse_args()

    return args

def run(args, img_path,sentiment_type, sentiment_scale,text_style_scale,text_to_mimic,desired_style_embedding_vector,cuda_idx,title2print,model_path):
    text_generator = CLIPTextGenerator(cuda_idx=cuda_idx,model_path = model_path, **vars(args))

    image_features = text_generator.get_img_feature([img_path], None)
    # SENTIMENT: added scale parameter
    captions = text_generator.run(image_features, args.cond_text, args.beam_size,sentiment_type,sentiment_scale,text_style_scale,text_to_mimic,desired_style_embedding_vector,cuda_idx)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print(title2print)
    print('best clip:', args.cond_text + captions[best_clip_idx])


    
    img_dict[img_path][text_style_scale][label] = args.cond_text + captions[best_clip_idx]

def run_arithmetic(args, imgs_path, img_weights):
    text_generator = CLIPTextGenerator(**vars(args))

    image_features = text_generator.get_combined_feature(imgs_path, [], img_weights, None)
    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print('best clip:', args.cond_text + captions[best_clip_idx])

# SENTIMENT: writing results to file
def write_results(img_dict):
    with open('results.csv', 'w') as results_file:
        writer = csv.writer(results_file)
        for img in img_dict.keys():    
            writer.writerow([img])
            writer.writerow(['scale/sentiment', 'negative', 'positive', 'neutral','none'])        
            for scale in img_dict[img].keys():
                cur_row = [scale]
                for sentiment in img_dict[img][scale].keys():
                    cur_row.append(img_dict[img][scale][sentiment])
                writer.writerow(cur_row)

def write_results_of_text_style(img_dict, embedding_type,labels,reults_dir,style_type):
    if not os.path.isdir(reults_dir):
        os.makedirs(reults_dir)
    tgt_path = os.path.join(reults_dir,f'results_{style_type}_embedding_type_{embedding_type}.csv')
    print(f'Writing results into: {tgt_path}')
    with open(tgt_path, 'w') as results_file:
        writer = csv.writer(results_file)
        for img in img_dict.keys():
            img_num_str = img.split('/')[-1].split('.j')[0]
            writer.writerow([img_num_str])
            titles = ['scale/label']
            titles.extend(labels)
            writer.writerow(titles)
            for scale in img_dict[img].keys():
                cur_row = [scale]
                for label in img_dict[img][scale].keys():
                    cur_row.append(img_dict[img][scale][label])
                writer.writerow(cur_row)


# SENTIMENT: running the model for each image, sentiment and sentiment-scale
if __name__ == "__main__":
    # twitter: 'BillGates', 'rihanna', 'justinbieber', 'JLo', 'elonmusk', 'KendallJenner'
    cuda_idx = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
    args = get_args()
 
    img_path_list = [105,104,103,102,101,100] # list(np.arange(100,105))
    sentiment_list = ['none']#['negative','positive','neutral', 'none']
    sentiment_scale_list = [2.0]#[2.0, 1.5, 1.0, 0.5, 0.1]
    base_path = '/home/bdaniela/zero-shot-style'
    text_style_scale_list = [0.5,1,2,4,8]#[3.0]

    text_to_mimic_list = ["I love you honey!!!"," I hate you and I want to kill you", "Let's set a meeting at work","Please say it formal","Please describe it angrily"]
    img_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: "")))
    # embedding_path_idx2str = {0:'mean',1:'median'}
    embedding_path_idx2str = {0:'mean'}
    # style_type = 'emotions'
    # style_type = 'twitter'
    # style_type_list = ['twitter','emotions']#todo remove comment
    style_type_list = ['emotions']
    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print(f'Cur time is: {cur_time}')
    for style_type in style_type_list:
        if style_type == 'emotions':
            model_path = os.path.join(base_path,'checkpoints','best_model','best_28_classes_trained_model_emotions.pth')
            embedding_path1 = os.path.join(base_path,'checkpoints','best_model', 'emotions_mean_class_embedding.p')
            embedding_path2 = os.path.join(base_path,'checkpoints','best_model', 'emotions_median_class_embedding.p')
            # desired_labels_list = ['gratitude', 'anger'] - need to be good partition
            desired_labels_list = 'all'
        elif style_type == 'twitter':
            model_path = os.path.join(base_path,'checkpoints','best_model',
                                      'best_twitter_trained_model_emotions.pth')
            embedding_path1 = os.path.join(base_path,'checkpoints','best_model',
                                           'twitter_mean_class_embedding.p')  # twitter
            embedding_path2 = os.path.join(base_path,'checkpoints','best_model',
                                           'twitter_median_class_embedding.p')  # twitter
            desired_labels_list = ['BillGates', 'rihanna', 'justinbieber']
        embedding_path_list = [embedding_path1, embedding_path2]
        for embedding_path_idx,embedding_path in enumerate(embedding_path_list):
            with open(embedding_path, 'rb') as fp:
                embedding_vectors_to_load = pickle.load(fp)  # mean_embedding_vectors_to_load = {'love': mean_love_embedding, 'anger': mean_anger_embedding}
            desired_labels_list = list(embedding_vectors_to_load.keys())
            img_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: "")))
            for i in img_path_list:  # img_path_list:
                args.caption_img_path = os.path.join(base_path, 'data', 'imgs',
                                                     str(i) + ".jpeg")  # "imgs/"+str(i)+".jpg"
                # reults_dir = os.path.join('/home/bdaniela/zero-shot-style/zero_shot_style/results',str(i))
                reults_dir = os.path.join(base_path, 'results', cur_time)
                if not os.path.isfile(args.caption_img_path):
                    continue
                for label in desired_labels_list:
                    desired_style_embedding_vector = embedding_vectors_to_load[label]
                    for s, sentiment_scale in enumerate(sentiment_scale_list):
                        for text_style_scale in text_style_scale_list:
                            for sentiment_type in sentiment_list:

                                if sentiment_type=='none' and s>0:
                                    continue

                                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                                title2print = f'~~~~~~~~\n{dt_string} | Work on img path: {args.caption_img_path} with:' \
                                              f'\nstyle_type= *** {style_type} ***' \
                                              f'\nstyle of: *** {label} ***\ntext_style_scale= *** {text_style_scale} ***' \
                                              f'\n embedding_type=*** {embedding_path_idx2str[embedding_path_idx]} ***.' \
                                              f'\n~~~~~~~~'
                                print(title2print)

                                if args.run_type == 'caption':
                                    run(args, args.caption_img_path, sentiment_type, sentiment_scale, text_style_scale, text_to_mimic, desired_style_embedding_vector, cuda_idx,title2print,model_path)
                                    # write_results(img_dict)
                                    write_results_of_text_style(img_dict,embedding_path_idx2str[embedding_path_idx],desired_labels_list,reults_dir,style_type)
                                elif args.run_type == 'arithmetics':
                                    args.arithmetics_weights = [float(x) for x in args.arithmetics_weights]
                                    run_arithmetic(args, imgs_path=args.arithmetics_imgs, img_weights=args.arithmetics_weights)
                                else:
                                    raise Exception('run_type must be caption or arithmetics!')

    print('Finish of program!')