import argparse
import torch
import clip
from zero_shot_style.model.ZeroCLIP import CLIPTextGenerator
from datetime import datetime
import os.path
import csv
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    #parser.add_argument("--cond_text", type=str, default="")
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

def run(args, img_path,sentiment_type, sentiment_scale,text_style_scale,text_to_mimic,embedding_path,desired_class):
    text_generator = CLIPTextGenerator(**vars(args))

    image_features = text_generator.get_img_feature([img_path], None)
    # SENTIMENT: added scale parameter
    captions = text_generator.run(image_features, args.cond_text, args.beam_size,sentiment_type,sentiment_scale,text_style_scale,text_to_mimic,embedding_path,desired_class)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print('best clip:', args.cond_text + captions[best_clip_idx])

    
    img_dict[img_path][sentiment_scale][sentiment_type] = args.cond_text + captions[best_clip_idx]

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
                writer.writerow([])

# SENTIMENT: running the model for each image, sentiment and sentiment-scale
if __name__ == "__main__":
    args = get_args()
 
    img_path_list = [33]#range(45)
    sentiment_list   = ['none']#['negative','positive','neutral', 'none']
    sentiment_scale_list = [2.0]#[2.0, 1.5, 1.0, 0.5, 0.1]
    base_path = '/home/bdaniela/zero-shot-style/zero_shot_style/model/data/imgs'
    text_style_scale_list = [3.0]
    # text_to_mimic_list = ["Oh my gosh, I don't believe it, It is amazing!!!",
    #                  "Today we are going to win and sell this product in million dollar.",
    #                  " BLA BLA BLA BLA"]

    text_to_mimic_list = ["I so like this party!!!",
                          "I succeed to do my business."]
    embedding_path = os.path.join(base_path, 'mean_class_embedding.p')
    img_dict = defaultdict(lambda: defaultdict(lambda :defaultdict(lambda: "")))
    desired_class = 'love'#anger

    for s, sentiment_scale in enumerate(sentiment_scale_list):
        for text_style_scale in text_style_scale_list:
            for text_to_mimic in text_to_mimic_list:
                for i in [33]:#img_path_list:
                    args.caption_img_path = os.path.join(base_path,str(i)+".jpg")#"imgs/"+str(i)+".jpg"
                    if not os.path.isfile(args.caption_img_path):
                        continue

                    for sentiment_type in sentiment_list:

                        if sentiment_type=='none' and s>0:
                            continue

                        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        print(f'~~~~~~~~\n{dt_string} | Work on img path: {args.caption_img_path} with:\n ***{sentiment_type}***  sentiment, sentiment scale=***{sentiment_scale}***'
                              f'\n text_style_scale=***{text_style_scale}*** with style of: ***{text_to_mimic}***.\n~~~~~~~~')

                        if args.run_type == 'caption':
                            run(args, args.caption_img_path, sentiment_type, sentiment_scale,text_style_scale,text_to_mimic,embedding_path,desired_class)
                            write_results(img_dict)
                        elif args.run_type == 'arithmetics':
                            args.arithmetics_weights = [float(x) for x in args.arithmetics_weights]
                            run_arithmetic(args, imgs_path=args.arithmetics_imgs, img_weights=args.arithmetics_weights)
                        else:
                            raise Exception('run_type must be caption or arithmetics!')

