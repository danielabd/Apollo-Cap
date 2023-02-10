import argparse

import pandas as pd
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
    parser.add_argument("--cond_text_list", nargs="+", type=str, default=["Image of a"])
    #parser.add_argument("--cond_text", type=str, default="Image of a")
    #parser.add_argument("--cond_text_list", nargs="+", type=str, default=[""])
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

    # parser.add_argument("--caption_img_dict", type=str, default=[os.path.join(os.path.expanduser('~'),'data','senticap'),
    #                                                                           os.path.join(os.path.expanduser('~'),
    #                                                                                        'data', 'flickrstyle10k')],
    #                     help="Path to images dict for captioning")
    parser.add_argument("--caption_img_dict", type=str,
                        default=[os.path.join(os.path.expanduser('~'), 'data', 'senticap')],
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



def write_data_to_global_file_for_debug(data, img_idx_to_name, tgt_results_path, t):
    with open(tgt_results_path, 'w') as results_file:
        writer = csv.writer(results_file)
        title = ['idx']
        title.extend(list(data.columns))
        writer.writerow(title)
        for i in img_idx_to_name.keys():
            cur_row = [i, img_idx_to_name[i]]
            if img_idx_to_name[i] in list(data[t]):
                idx = list(data[t]).index(img_idx_to_name[i])
                cur_row.extend(list(data.iloc[idx].values[1:]))
            writer.writerow(cur_row)
    print(f'Finished to write data to global file for debug in: {tgt_results_path}')

def merge_res_files_to_one(exp_to_merge,  res_paths,  src_dirs, t, tgt_paths, factual_wo_prompt):
    #go over test type
    keys_test_type = {}
    factual_image_of_a_prompt_manipulation = {}
    factual_image_manipulation = {}
    factual_image_of_a_image_and_prompt_manipulation = {}
    for test_type in exp_to_merge:
        total_data = {}
        # go over all dirs of this test type
        for d in res_paths[test_type]:
            if d.startswith(".") or os.path.isfile(os.path.join(src_dirs[test_type],d)):
                continue
            files = os.listdir(os.path.join(src_dirs[test_type],d))
            # take the relevant file
            for f in files:
                if test_type == 'text_style':
                    if f != "results_23_26_35__05_02_2023.csv":
                        continue
                if f.endswith('.csv'):
                    path_file = os.path.join(src_dirs[test_type],d,f)
                    break

            data = pd.read_csv(path_file)
            if f!='results_all_models_source_classes_03_43_42__10_02_2023.csv':
                data = data.head(data.shape[0] - 1) #remove last line for the case that it is not completed

            for i,k in enumerate(data[t[test_type]]):
                pos = data['positive'][i]
                try:
                    neg = data['negative'][i]
                except:
                    neg = data['ngeative'][i]
                try:
                    fact = data['factual'][i]
                except:
                    try:
                        fact = data['neutral'][i]
                    except:
                        fact = ''
                if test_type == 'prompt_manipulation':
                    factual_image_of_a_prompt_manipulation[k] = fact
                # if test_type == 'image_and_prompt_manipulation':
                #     factual_image_of_a_image_and_prompt_manipulation[k] = fact
                #     if fact!= factual_image_of_a_prompt_manipulation[k]:
                #         print('check')
                if test_type != 'prompt_manipulation':
                    fact = factual_image_of_a_prompt_manipulation[k]
                single_data = {'img_num': k, 'factual': fact, 'positive': pos, 'negative': neg}
                total_data[k] = single_data
        keys_test_type[test_type] = list(total_data.keys())
        total_data_test_type = pd.DataFrame(list(total_data.values()))
        for i in list(total_data.values()):
            if i['img_num'] == 104906:
                print("here")
        total_data_test_type.to_csv(tgt_paths[test_type], index=False, header=True)
        print(f"finish to create: {tgt_paths[test_type]}")
        # write_data_to_global_file_for_debug(total_data_test_type, img_idx_to_name, tgt_paths_debug[test_type], t[test_type])
        # for
    print('Finish of program!')
    imgs_to_add = {}
    for test_type in  keys_test_type:
        if test_type=='text_style':
            continue
        else:
            imgs_to_add[test_type] = []
        for i in keys_test_type['text_style']:
            if i not in keys_test_type[test_type]:
                imgs_to_add[test_type].append(i)
        print(f"For {test_type}, need to add images: ")
        print(imgs_to_add[test_type])

    print('Finish of program!')



def get_all_paths(cur_time, factual_wo_prompt):
    #prompt_manipulation
    src_dir_prompt_manipulation = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/prompt_manipulation'
    prompt_manipulation_dir_path = os.listdir(src_dir_prompt_manipulation)
    if factual_wo_prompt:
        tgt_path_prompt_manipulation = os.path.join(src_dir_prompt_manipulation,'total_results_prompt_manipulation_factual_wo_prompt.csv')
    else:
        tgt_path_prompt_manipulation = os.path.join(src_dir_prompt_manipulation,'total_results_prompt_manipulation.csv')

    #image and prompt_manipulation
    src_dir_image_and_prompt_manipulation = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/image_and_prompt_manipulation'
    image_and_prompt_manipulation_dir_path = os.listdir(src_dir_image_and_prompt_manipulation)
    if factual_wo_prompt:
        tgt_path_image_and_prompt_manipulation = os.path.join(src_dir_image_and_prompt_manipulation,'total_results_image_and_prompt_manipulation_factual_wo_prompt.csv')
    else:
        tgt_path_image_and_prompt_manipulation = os.path.join(src_dir_image_and_prompt_manipulation,'total_results_image_and_prompt_manipulation.csv')

    #text style
    src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/text_style'
    text_style_dir_path = os.listdir(src_dir_text_style)
    if factual_wo_prompt:
        tgt_path_text_style = os.path.join(src_dir_text_style,'total_results_text_style_factual_wo_prompt.csv')
    else:
        tgt_path_text_style = os.path.join(src_dir_text_style,'total_results_text_style.csv')

    # image manipulation
    src_dir_image_manipulation = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/image_manipulation'
    image_manipulation_dir_path = os.listdir(src_dir_image_manipulation)
    if factual_wo_prompt:
        tgt_path_image_manipulation = os.path.join(src_dir_image_manipulation,'total_results_image_manipulation_factual_wo_prompt.csv')
    else:
        tgt_path_image_manipulation = os.path.join(src_dir_image_manipulation,'total_results_image_manipulation.csv')

    debug_tgt_path_im_manipulation = os.path.join(os.path.expanduser('~'), 'results', cur_time+'_debug_total_results_image_manipulation.csv')
    debug_tgt_path_prompt_manipulation = os.path.join(os.path.expanduser('~'), 'results',
                                                cur_time+'_debug_total_results_prompt_manipulation.csv')

    src_dirs = {"prompt_manipulation":  src_dir_prompt_manipulation, "image_manipulation": src_dir_image_manipulation, "image_and_prompt_manipulation": src_dir_image_and_prompt_manipulation, "text_style": src_dir_text_style}
    res_paths = {"prompt_manipulation": prompt_manipulation_dir_path, "image_manipulation": image_manipulation_dir_path, "image_and_prompt_manipulation": image_and_prompt_manipulation_dir_path, "text_style":text_style_dir_path}
    tgt_paths = {"prompt_manipulation": tgt_path_prompt_manipulation,"image_manipulation": tgt_path_image_manipulation, "image_and_prompt_manipulation": tgt_path_image_and_prompt_manipulation, "text_style": tgt_path_text_style}
    tgt_paths_debug = {"prompt_manipulation": debug_tgt_path_prompt_manipulation,"image_manipulation": debug_tgt_path_im_manipulation, "image_and_prompt_manipulation": "debug_tgt_path_image_and_prompt_manipulation"}

    return res_paths, src_dirs, tgt_paths


def main():
    args = get_args()
    cuda_idx = args.cuda_idx_num
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx

    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print(f'Cur time is: {cur_time}')

    imgs_to_test = []

    for setdir in args.caption_img_dict:
        print(f'setdir={setdir}')
        for im in os.listdir(os.path.join(setdir,'images','test')):
            if ('.jpg' or '.jpeg' or '.png') not in im:
                continue
            imgs_to_test.append(os.path.join(setdir,'images','test',im))

    tgt_results_path = os.path.join(os.path.expanduser('~'), 'results', "img_idx_to_name.csv")
    img_idx_to_name = {}
    for img_path_idx, img_path in enumerate(imgs_to_test):  # img_path_list:
        img_name = img_path.split('/')[-1].split('.jpg')[0]
        try:
            img_name = str(int(img_name))
        except:
            pass
        img_idx_to_name[img_path_idx] = img_name

    t = {"prompt_manipulation": "img_num\prompt","image_manipulation": "img_num\style",  "image_and_prompt_manipulation": "img_num\style", "text_style": "img_num"}
    factual_wo_prompt = True
    res_paths, src_dirs, tgt_paths = get_all_paths(cur_time, factual_wo_prompt)

    exp_to_merge = ["prompt_manipulation", "image_and_prompt_manipulation", "image_manipulation", "text_style"]
    # exp_to_merge = ["prompt_manipulation", "image_and_prompt_manipulation", "text_style"]

    merge_res_files_to_one(exp_to_merge, res_paths, src_dirs, t, tgt_paths, factual_wo_prompt)
    print("finish program")


if __name__ == "__main__":
    main()
