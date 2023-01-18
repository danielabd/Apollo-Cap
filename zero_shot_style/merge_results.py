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

    global_results_dir_path = os.path.join(os.path.expanduser('~'),'results')
    # prompt_manipulation_dir_path = ['01_06_14__27_12_2022','00_59_38__27_12_2022','00_56_45__27_12_2022','14_14_09__23_12_2022','14_15_04__23_12_2022','12_47_53__22_12_2022','01_11_22__29_12_2022', '01_14_18__29_12_2022', '01_17_10__29_12_2022', '01_51_19__29_12_2022']
    # image_manipulation_dir_path = ['01_07_14__27_12_2022','00_58_25__27_12_2022','00_54_50__27_12_2022','14_14_43__23_12_2022','14_15_55__23_12_2022','12_48_26__22_12_2022']
    prompt_manipulation_dir_path = ['21_57_03__02_01_2023','21_57_57__02_01_2023','21_58_25__02_01_2023','02_03_43__05_01_2023','21_57_57__02_01_2023','02_06_18__05_01_2023','13_39_30__05_01_2023','21_58_25__02_01_2023',]
    image_manipulation_dir_path = []
    tgt_path_im_manipulation = os.path.join(os.path.expanduser('~'),'results',cur_time+'_total_results_image_manipulation.csv')
    tgt_path_prompt_manipulation = os.path.join(os.path.expanduser('~'),'results',cur_time+'_total_results_prompt_manipulation.csv')
    debug_tgt_path_im_manipulation = os.path.join(os.path.expanduser('~'), 'results', cur_time+'_debug_total_results_image_manipulation.csv')
    debug_tgt_path_prompt_manipulation = os.path.join(os.path.expanduser('~'), 'results',
                                                cur_time+'_debug_total_results_prompt_manipulation.csv')
    res_paths = {"prompt_manipulation": prompt_manipulation_dir_path, "im_manipulation": image_manipulation_dir_path}
    tgt_paths = {"prompt_manipulation": tgt_path_prompt_manipulation,"im_manipulation": tgt_path_im_manipulation}
    tgt_paths_debug = {"prompt_manipulation": debug_tgt_path_prompt_manipulation,"im_manipulation": debug_tgt_path_im_manipulation}
    t = {"prompt_manipulation": "img_num\prompt","im_manipulation": "img_num\style"}
    #exp_to_merge = ["prompt_manipulation", "im_manipulation"]
    exp_to_merge = ["prompt_manipulation"]

    for test_type in exp_to_merge:
        total_data_test_type = pd.DataFrame()
        for d in res_paths[test_type]:
            path_file = os.path.join(global_results_dir_path,d,f'results_all_models_source_classes_{d}.csv')
            data = pd.read_csv(path_file)
            data = data.head(data.shape[0] - 1)
            for i,k1 in enumerate(data[t[test_type]]):
                k = k1.split('.jpg')[0]
                try:
                    k = str(int(k))
                except:
                    pass
                data[t[test_type]][i] = k
            total_data_test_type = pd.concat([total_data_test_type, data])
        total_data_test_type.to_csv(tgt_paths[test_type], index=False, header=True)
        write_data_to_global_file_for_debug(total_data_test_type, img_idx_to_name, tgt_paths_debug[test_type], t[test_type])

    print('Finish of program!')

if __name__ == "__main__":
    main()
