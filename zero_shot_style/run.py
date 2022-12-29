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
from utils import parser, get_hparams

def get_args():
    #parser = argparse.ArgumentParser() #comment when using, in addition, the arguments from zero_shot_style.utils
    parser.add_argument("--img_name", type=int, default=0)
    parser.add_argument("--use_all_imgs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    #parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo or gpt-j")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    #parser.add_argument("--cond_text_list", nargs="+", type=str, default=["Image of a"])
    #parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--cond_text_list", nargs="+", type=str, default=["I am an intelligent image captioning bot. This image is a {img_type}. There {num_people}. I think this photo was taken at a {place1}, {place2}, or {place3}. I think there might be a {object1}, {object2}, {object3},... in this {img_type}. A creative short caption I can generate to describe this image is:"])
    parser.add_argument("--cond_text", type=str, default="")
    parser.add_argument("--cond_text2", type=str, default="")
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

    parser.add_argument("--cuda_idx_num", type=str, default="1")
    parser.add_argument("--img_idx_to_start_from", type=int, default=0)

    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'arithmetics'])

    parser.add_argument("--caption_img_dict", type=str, default=[os.path.join(os.path.expanduser('~'),'data','senticap'),
                                                                              os.path.join(os.path.expanduser('~'),
                                                                                           'data', 'flickrstyle10k')],
                        help="Path to images dict for captioning")
    '''
    parser.add_argument("--caption_img_dict", type=str, default=[os.path.join(os.path.expanduser('~'),'data','imgs')],
                        help="Path to images dict for captioning")
    '''
    parser.add_argument("--caption_img_path", type=str, default=os.path.join(os.path.expanduser('~'),'data','imgs','101.jpeg'),
                        help="Path to image for captioning")

    parser.add_argument("--arithmetics_imgs", nargs="+",
                        default=['example_images/arithmetics/woman2.jpg',
                                 'example_images/arithmetics/king2.jpg',
                                 'example_images/arithmetics/man2.jpg'])

    parser.add_argument("--arithmetics_style_imgs", nargs="+",
                        default=['49','50','51','52','53'])
    parser.add_argument("--arithmetics_weights", nargs="+", default=[1, 1, -1])
    parser.add_argument("--use_style_model", type=bool, default=False)

    args = parser.parse_args()

    return args

def run(args, img_path,sentiment_type, sentiment_scale,text_style_scale,imitate_text_style,desired_style_embedding_vector,cuda_idx,title2print,model_path,style_type,tmp_text_loss,label,img_dict):
    text_generator = CLIPTextGenerator(cuda_idx=cuda_idx,model_path = model_path,tmp_text_loss= tmp_text_loss, **vars(args))

    image_features = text_generator.get_img_feature([img_path], None)
    # SENTIMENT: added scale parameter
    if imitate_text_style:
        text_style = label
    else:
        text_style = ''
    captions = text_generator.run(image_features, args.cond_text, args.beam_size,sentiment_type,sentiment_scale,text_style_scale,text_style,desired_style_embedding_vector,style_type)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    new_title2print = f'~~~~~~~~\n{dt_string} | Work on img path:' + title2print.split(' | Work on img path:')[1]
    print(new_title2print)

    print('best clip:', args.cond_text + captions[best_clip_idx])

    img_dict[img_path][style_type][text_style_scale][label] = args.cond_text + captions[best_clip_idx]

def run_arithmetic(args, img_dict_img_arithmetic,base_img,style_type, imgs_path, img_weights, cuda_idx,title2print):
    #text_generator = CLIPTextGenerator(**vars(args))
    text_generator = CLIPTextGenerator(cuda_idx=cuda_idx, **vars(args))

    image_features = text_generator.get_combined_feature(imgs_path, [], img_weights, None)
    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    new_title2print = f'~~~~~~~~\n{dt_string} | Work on img path:' + title2print.split(' | Work on img path:')[1]
    print(new_title2print)

    print('best clip:', args.cond_text + captions[best_clip_idx])
    img_dict_img_arithmetic[base_img][style_type] = args.cond_text + captions[best_clip_idx]


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

def write_results_of_text_style_all_models(img_dict,labels,reults_dir,scales_len,tgt_results_path):
    # img_dict[img_path][style_type][text_style_scale][label]
    if not os.path.isdir(reults_dir):
        os.makedirs(reults_dir)
    print(f'Writing results into: {tgt_results_path}')
    with open(tgt_results_path, 'w') as results_file:
        writer = csv.writer(results_file)
        for img in img_dict.keys():
            img_num_str = img.split('/')[-1].split('.j')[0]
            titles0 = ['img_num']
            titles0.extend([img_num_str] * scales_len*len(labels))
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


def write_results_prompt_manipulation(img_dict,labels,reults_dir,scales_len,tgt_results_path):
    # img_dict[img_path][style_type][text_style_scale][label]
    if not os.path.isdir(reults_dir):
        os.makedirs(reults_dir)
    print(f'Writing results into: {tgt_results_path}')
    writeTitle = True
    with open(tgt_results_path, 'w') as results_file:
        writer = csv.writer(results_file)
        for i,img in enumerate(img_dict.keys()):
            img_num_str = img.split('/')[-1].split('.j')[0]
            cur_row = [img_num_str]
            for model_name in img_dict[img]:
                for scale in img_dict[img][model_name].keys():
                    labels = img_dict[img][model_name][scale].keys()
                    if writeTitle:
                        titles0 = ['img_num\prompt']
                        titles0.extend(labels)
                        writer.writerow(titles0)
                        writeTitle = False
                    for label in labels:
                        cur_row.append(img_dict[img][model_name][scale][label])
            writer.writerow(cur_row)

def write_img_idx_to_name(img_idx_to_name, tgt_results_path):
    print(f'Writing img idx to name: {tgt_results_path}..')
    with open(tgt_results_path, 'w') as results_file:
        writer = csv.writer(results_file)
        for i in img_idx_to_name.keys():
            cur_row = [i, img_idx_to_name[i]]
            writer.writerow(cur_row)
    print(f'Finished to write img idx to name: {tgt_results_path}')


def write_results_image_manipulation(img_dict_img_arithmetic, labels,reults_dir,scales_len,tgt_results_path,imgs_style_type_dict):
    if not os.path.isdir(reults_dir):
        os.makedirs(reults_dir)
    print(f'Writing results into: {tgt_results_path}')
    writeTitle = True
    with open(tgt_results_path, 'w') as results_file:
        writer = csv.writer(results_file)
        for i, img in enumerate(img_dict_img_arithmetic.keys()):
            img_num_str = img.split('/')[-1].split('.j')[0]
            cur_row = [img_num_str]
            styles = img_dict_img_arithmetic[img].keys()
            if writeTitle:
                titles0 = ['img_num\style']
                titles0.extend(styles)
                writer.writerow(titles0)
                writeTitle = False
            for style_type in styles:
                cur_row.append(img_dict_img_arithmetic[img][style_type])
            writer.writerow(cur_row)


def get_title2print(caption_img_path, style_type, label, text_style_scale, embedding_path_idx):
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    title2print = f'~~~~~~~~\n{dt_string} | Work on img path: {caption_img_path} with:' \
                  f'\nstyle_type= *** {style_type} ***' \
                  f'\nstyle of: *** {label} ***\ntext_style_scale= *** {text_style_scale} ***' \
                  f'\n embedding_type=*** {embedding_path_idx} ***.' \
                  f'\n~~~~~~~~'
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


def main():
    #cuda_idx = "1"
    #os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
    args = get_args()
    config = get_hparams(args)
    imgs_style_type_dict = {49: 'neutral', 50:'positive', 51:'negative', 52:'humor', 53:'romantic'}

    if not args.img_name:
        img_path_list = list(np.arange(0,20000))#[35]#[101, 105, 104, 103, 102, 100]  # list(np.arange(100,105))
    else:
        img_path_list = [args.img_name]

    cuda_idx = args.cuda_idx_num
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx

    sentiment_list = ['none']  # ['negative','positive','neutral', 'none']
    sentiment_scale_list = [2.0]  # [2.0, 1.5, 1.0, 0.5, 0.1]
    base_path = os.path.join(os.path.expanduser('~'), 'projects','zero-shot-style')
    checkpoints_dir = os.path.join(os.path.expanduser('~'), 'checkpoints')
    data_dir = os.path.join(os.path.expanduser('~'), 'data')
    text_style_scale_list = [4]  # [0,0.5,1,2,4,8]#[0,1,2,4,8]#[0.5,1,2,4,8]#[3.0]#
    text_to_imitate_list = ["bla"]#["Happy", "Love", "angry", "hungry", "I love you!!!", " I hate you and I want to kill you",
                            #"Let's set a meeting at work", "I angry and I love", "The government is good"]
    imitate_text_style = False
    embedding_path_idx2str = {0: 'mean'}

    style_type_list = ['twitter']  # ['clip','twitter','emotions']#['emotions_love_disgust']

    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print(f'Cur time is: {cur_time}')
    img_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ""))))
    img_dict_img_arithmetic = defaultdict(lambda: defaultdict(lambda: "")) #img_path,style_type
    tmp_text_loss = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: "")))
    #for imitate_text_style in [False]:
    if imitate_text_style:
        classes_type = "sentences"
    else:
        classes_type = "source"

    imgs_to_test = []

    for setdir in args.caption_img_dict:
        print(f'setdir={setdir}')
        for im in os.listdir(os.path.join(setdir,'images','test')):
            if ('.jpg' or '.jpeg' or '.png') not in im:
                continue
            imgs_to_test.append(os.path.join(setdir,'images','test',im))
    #imgs_to_test = [args.caption_img_path] #for test one image

    '''
    tgt_results_path = os.path.join(os.path.expanduser('~'), 'results', "img_idx_to_name.csv")
    img_idx_to_name = {}
    for img_path_idx, img_path in enumerate(imgs_to_test):  # img_path_list:
        img_name = img_path.split('/')[-1]
        img_idx_to_name[img_path_idx] = img_name
    write_img_idx_to_name(img_idx_to_name, tgt_results_path)
    exit(0)
    '''

    for img_path_idx, img_path in enumerate(imgs_to_test):  # img_path_list:
        if img_path_idx < args.img_idx_to_start_from:
            continue
        #args.caption_img_path = get_img_full_path(base_path,img_idx)
        #args.caption_img_path = os.path.join(os.path.expanduser('~'),'data','flickrstyle10k','flickr_images_dataset',img_name)
        #if not args.caption_img_path:
        #    continue
        img_name = img_path.split('/')[-1]
        args.caption_img_path = img_path
        for prompt in args.cond_text_list:
            args.cond_text = prompt
            #reults_dir = os.path.join(base_path, 'results', cur_time)
            reults_dir = os.path.join(os.path.expanduser('~'), 'results', cur_time)
            tgt_results_path = os.path.join(reults_dir, f'results_all_models_{classes_type}_classes_{cur_time}.csv')

            if not os.path.isfile(args.caption_img_path):
                continue
            model_path = os.path.join(checkpoints_dir, 'best_model',
                                      config['best_model_name'])
            mean_embedding_vec_path = os.path.join(checkpoints_dir, 'best_model',
                                                   config['mean_vec_emb_file'])
            median_embedding_vec_path = os.path.join(checkpoints_dir, 'best_model',
                                                     config['median_vec_emb_file'])
            desired_labels_list = config['desired_labels']
            for style_type in style_type_list:
                embedding_path_list = [mean_embedding_vec_path]
                for embedding_path_idx, embedding_path in enumerate(embedding_path_list):
                    if args.use_style_model:
                        with open(embedding_path, 'rb') as fp:
                            embedding_vectors_to_load = pickle.load(fp)
                        desired_labels_list = list(embedding_vectors_to_load.keys())
                    else:
                        desired_labels_list = [prompt]
                    if imitate_text_style:
                        desired_labels_list = text_to_imitate_list
                    for label in desired_labels_list:
                        desired_style_embedding_vector = ''
                        if not imitate_text_style:
                            if args.use_style_model:
                                desired_style_embedding_vector = embedding_vectors_to_load[label]
                        for s, sentiment_scale in enumerate(sentiment_scale_list):
                            for text_style_scale_idx, text_style_scale in enumerate(text_style_scale_list):
                                for sentiment_type in sentiment_list:
                                    if sentiment_type == 'none' and s > 0:
                                        continue


                                    if args.run_type == 'caption':
                                        pass
                                        title2print = get_title2print(args.caption_img_path, style_type, label,
                                                                      text_style_scale,
                                                                      embedding_path_idx2str[embedding_path_idx])
                                        print(title2print)
                                        run(args, args.caption_img_path, sentiment_type, sentiment_scale,
                                            text_style_scale, imitate_text_style, desired_style_embedding_vector,
                                            cuda_idx, title2print, model_path, style_type,tmp_text_loss,label,img_dict)
                                        if not args.use_style_model:
                                            write_results_prompt_manipulation(img_dict, desired_labels_list,
                                                                               reults_dir,
                                                                               len(text_style_scale_list),
                                                                               tgt_results_path)
                                        # # write_results_of_text_style(img_dict,embedding_path_idx2str[embedding_path_idx],desired_labels_list,reults_dir,style_type)
                                        if args.use_style_model:
                                            write_results_of_text_style_all_models(img_dict, desired_labels_list,
                                                                               reults_dir,
                                                                               len(text_style_scale_list),
                                                                               tgt_results_path)
                                    elif args.run_type == 'arithmetics':
                                        #none arithmetic
                                        title2print = get_title2print(args.caption_img_path, style_type, 'neutral',
                                                                      text_style_scale,
                                                                      embedding_path_idx2str[embedding_path_idx])
                                        print(title2print)
                                        args.arithmetics_imgs = [args.caption_img_path, args.caption_img_path, args.caption_img_path]
                                        run_arithmetic(args, img_dict_img_arithmetic, img_name,
                                                       'none', imgs_path=args.arithmetics_imgs,
                                                       img_weights=[1, 0, 0], cuda_idx=cuda_idx,title2print = title2print)
                                        write_results_image_manipulation(img_dict_img_arithmetic, desired_labels_list,
                                                                         reults_dir,
                                                                         len(text_style_scale_list),
                                                                         tgt_results_path, imgs_style_type_dict)

                                        args.arithmetics_weights = [float(x) for x in args.arithmetics_weights]
                                        neutral_img_style = get_full_path_of_stylized_images(data_dir, args.arithmetics_style_imgs[0])
                                        for idx, v in enumerate(args.arithmetics_style_imgs[1:]):
                                            img_style = get_full_path_of_stylized_images(data_dir, v)
                                            args.arithmetics_imgs = [args.caption_img_path, neutral_img_style, img_style]

                                            title2print = get_title2print(args.caption_img_path, style_type, imgs_style_type_dict[int(v)],
                                                                          text_style_scale,
                                                                          embedding_path_idx2str[embedding_path_idx])

                                            run_arithmetic(args,img_dict_img_arithmetic,img_name,imgs_style_type_dict[int(v)], imgs_path=args.arithmetics_imgs,
                                                           img_weights=args.arithmetics_weights, cuda_idx=cuda_idx,title2print = title2print)
                                            write_results_image_manipulation(img_dict_img_arithmetic, desired_labels_list,
                                                                              reults_dir,
                                                                              len(text_style_scale_list),
                                                                              tgt_results_path,imgs_style_type_dict)

                                    else:
                                        raise Exception('run_type must be caption or arithmetics!')

    print('Finish of program!')

if __name__ == "__main__":
    main()
