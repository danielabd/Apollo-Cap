import math
import pandas as pd
import os.path
import csv
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

def merge_list_res_files_to_one(file_list, tgt_path):
    #go over test type
    total_data = {}
    for f in file_list:
        data = pd.read_csv(f)
        if not isinstance(data.iloc[-1,-1], str) and math.isnan(data.iloc[-1, -1]) or len(data.iloc[-1, :]) < data.shape[1]:
             data = data.head(data.shape[0] - 1)
        if 'img_num' in data:
            col_name = 'img_num'
        else:
            col_name = 'img_num\style'
        for i,k in enumerate(data[col_name]):
            try:
                pos = data['positive'][i]
            except:
                pos = None
            try:
                neg = data['negative'][i]
            except:
                neg = None
            total_data[k] = {'img_num': k, 'positive': pos, 'negative': neg}
        total_data_test_type = pd.DataFrame(list(total_data.values()))
        total_data_test_type.to_csv(tgt_path, index=False, header=True)
        print(f"finish to create: {tgt_path}")
    print('Finish of program!')


def get_results_of_single_folder(total_data, total_data_to_check, path_d, img_name_to_idx, use_factual=False, merge_sweep=False):
    sweep_exp = path_d.split('/')[-1]
    path_file = ''
    files = os.listdir(path_d)
    # take the relevant file
    for f in files:
        if f.startswith('avg_'):
            continue
        if f.endswith('.csv'):
            path_file = os.path.join(path_d,f)
            break
    if not path_file:
        return total_data, total_data_to_check
    data = pd.read_csv(path_file)
    #uncomment it for the cas we want to illuminate not comleted caption
    # if not isinstance(data.iloc[-1,-1], str) and math.isnan(data.iloc[-1, -1]) or len(data.iloc[-1, :]) < 3:
    #      data = data.head(data.shape[0] - 1)  # remove last line for the case that it is not completed
    label1=None; label2=None
    if 'positive' in data.columns:
        label1 = 'positive'
    elif 'humor' in data.columns:
        label1 = 'humor'
    if 'negative' in data.columns:
        label2 = 'negative'
    elif 'romantic' in data.columns:
        label2 = 'romantic'
    # for i,k in enumerate(data[data.columns[0]]):
    check = False
    for i,k in enumerate(data['img_num']):
        if label1:
            pos = data[label1][i]
            if len(pos.split(' ')) < 2:
                check = True
                # continue
        if label2:
            neg = data[label2][i]
            if len(neg.split(' ')) < 2:
                check = True
                # continue
        if use_factual:
            try:
                fact = data['factual'][i]
            except:
                try:
                    fact = data['neutral'][i]
                except:
                    fact = ''
            # if test_type == 'prompt_manipulation':
            #     factual_image_of_a_prompt_manipulation[k] = fact
            # if test_type == 'image_manipulation':
            #    image_manipulation factual_image_manipulation[k] = fact
            # if test_type == 'image_and_prompt_manipulation':
            #     factual_image_of_a_image_and_prompt_manipulation[k] = fact
            #     if fact!= factual_image_of_a_prompt_manipulation[k]:
            #         print('check')
            # if factual_wo_prompt:
            #     fact = factual_image_manipulation[k]
            # else: #write factual with prompt of image of a...
            #     fact = factual_image_of_a_prompt_manipulation[k]
            # single_data = {'img_num': k, 'factual': fact, label1: pos, label2: neg}

            # single_data = {'img_num': k, label1: pos, label2: neg}
            single_data = {'img_num': k}
        else:
            # single_data = {'idx': img_name_to_idx[k], 'img_num': k, label1: pos, label2: neg}
            # single_data = {'idx': img_name_to_idx[k], 'img_num': k,'sweep_exp': sweep_exp, label1: pos, label2: neg}
            if merge_sweep:
                single_data = {'idx': img_name_to_idx[k], 'img_num': k,'sweep_exp': sweep_exp}
            else:
                single_data = {'idx': img_name_to_idx[k], 'img_num': k}
        if label1:
            single_data[label1] = pos
        if label2:
            single_data[label2] = neg
        if merge_sweep:
            total_data[sweep_exp] = single_data
        else:
            if check:
                total_data_to_check[img_name_to_idx[k]] = single_data
            else:
                total_data[img_name_to_idx[k]] = single_data
            check = False
            # total_data[img_name_to_idx[k]] = single_data
    return total_data, total_data_to_check


def merge_res_files_to_one(exp_to_merge,  res_paths,  src_dirs, t, tgt_paths, factual_wo_prompt, use_factual,img_name_to_idx):
    #go over test type
    keys_test_type = {}

    if use_factual:
        factual_image_of_a_prompt_manipulation = {}
        factual_image_manipulation = {}
        if factual_wo_prompt:
            exp_to_merge.remove('image_manipulation')
            exp_to_merge.insert(0,'image_manipulation')
        else:
            exp_to_merge.remove('prompt_manipulation')
            exp_to_merge.insert(0, 'prompt_manipulation')
    for test_type in exp_to_merge:
        total_data = {}
        total_data_to_check = {}
        # go over all dirs of this test type
        dirs_of_res_paths = os.listdir(res_paths[test_type])
        for d in dirs_of_res_paths:
            path_d = os.path.join(res_paths[test_type],d)
            if os.path.isdir(path_d):
                total_data, total_data_to_check = get_results_of_single_folder(total_data,total_data_to_check, path_d,img_name_to_idx)
        for i in total_data_to_check:
            if i not in total_data:
                total_data[i] = total_data_to_check[i]
        keys_test_type[test_type] = list(total_data.keys())
        total_data_test_type = pd.DataFrame(list(total_data.values()))
        print(f"total keys in test = {len(total_data_test_type)}")
        for i in list(total_data.values()):
            if i['img_num'] == 104906:
                print("here")
        total_data_test_type.to_csv(tgt_paths[test_type], index=False, header=True)
        print(f"finish to create: {tgt_paths[test_type]}")
        # write_data_to_global_file_for_debug(total_data_test_type, img_idx_to_name, tgt_paths_debug[test_type], t[test_type])
        # for
    print('Finish of program!')

    # imgs_to_add = {}
    # for test_type in  keys_test_type:
    #     if test_type=='text_style':
    #         continue
    #     else:
    #         imgs_to_add[test_type] = []
    #     for i in keys_test_type['text_style']:
    #         if i not in keys_test_type[test_type]:
    #             imgs_to_add[test_type].append(i)
    #     print(f"For {test_type}, need to add images: ")
    #     print(imgs_to_add[test_type])

    print('Finish of program!')

def merge_res_sweep_to_one(exp_to_merge,  res_paths,  src_dirs, t, tgt_paths, factual_wo_prompt, use_factual,img_name_to_idx):
    #go over test type
    keys_test_type = {}

    if use_factual:
        factual_image_of_a_prompt_manipulation = {}
        factual_image_manipulation = {}
        if factual_wo_prompt:
            exp_to_merge.remove('image_manipulation')
            exp_to_merge.insert(0,'image_manipulation')
        else:
            exp_to_merge.remove('prompt_manipulation')
            exp_to_merge.insert(0, 'prompt_manipulation')
    for test_type in exp_to_merge:
        total_data = {}
        total_data_to_check = {}
        # go over all dirs of this test type
        dirs_of_res_paths = os.listdir(res_paths[test_type])
        for d in dirs_of_res_paths:
            path_d = os.path.join(res_paths[test_type],d)
            if os.path.isdir(path_d):
                total_data, total_data_to_check = get_results_of_single_folder(total_data,total_data_to_check, path_d,img_name_to_idx, merge_sweep=True)
        for i in total_data_to_check:
            if i not in total_data:
                total_data[i] = total_data_to_check[i]
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

    # imgs_to_add = {}
    # for test_type in  keys_test_type:
    #     if test_type=='text_style':
    #         continue
    #     else:
    #         imgs_to_add[test_type] = []
    #     for i in keys_test_type['text_style']:
    #         if i not in keys_test_type[test_type]:
    #             imgs_to_add[test_type].append(i)
    #     print(f"For {test_type}, need to add images: ")
    #     print(imgs_to_add[test_type])

    print('Finish of program!')


def bu_get_all_paths(cur_time, factual_wo_prompt, exp_to_merge,suffix_name):
    # exp_to_merge = ["prompt_manipulation", "image_and_prompt_manipulation", "image_manipulation", "text_style"]
    base_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/12_2_23/'
    base_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/23_2_23/'
    base_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/4_3_23/res_f_36'
    base_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/flickrstyle10k_fixed_param_25_3_23'

    # prompt_manipulation
    if 'prompt_manipulation' in exp_to_merge:
        # src_dir_prompt_manipulation = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/prompt_manipulation'
        src_dir_prompt_manipulation = os.path.join(base_path,'prompt_manipulation')
        prompt_manipulation_dir_path = os.listdir(src_dir_prompt_manipulation)
        if factual_wo_prompt:
            tgt_path_prompt_manipulation = os.path.join(src_dir_prompt_manipulation,'total_results_prompt_manipulation_factual_wo_prompt.csv')
        else:
            tgt_path_prompt_manipulation = os.path.join(src_dir_prompt_manipulation,'total_results_prompt_manipulation.csv')
    else:
        src_dir_prompt_manipulation = ''
        prompt_manipulation_dir_path = ''
        tgt_path_prompt_manipulation = ''

    #image and prompt_manipulation
    if 'image_and_prompt_manipulation' in exp_to_merge:
        src_dir_image_and_prompt_manipulation = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/image_and_prompt_manipulation'
        src_dir_image_and_prompt_manipulation = os.path.join(base_path,'image_and_prompt_manipulation')
        image_and_prompt_manipulation_dir_path = os.listdir(src_dir_image_and_prompt_manipulation)
        if factual_wo_prompt:
            tgt_path_image_and_prompt_manipulation = os.path.join(src_dir_image_and_prompt_manipulation,'total_results_image_and_prompt_manipulation_factual_wo_prompt.csv')
        else:
            tgt_path_image_and_prompt_manipulation = os.path.join(src_dir_image_and_prompt_manipulation,'total_results_image_and_prompt_manipulation.csv')
    else:
        src_dir_image_and_prompt_manipulation = ''
        image_and_prompt_manipulation_dir_path = ''
        tgt_path_image_and_prompt_manipulation = ''

    #text_style
    if 'text_style' in exp_to_merge:
        # src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/text_style'
        # 12.2.23
        # src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/12_2_23/text_style'
        # 20.2.23
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/20_2_23/ZeroStyleCap_8'
        # 23.2.23


        # src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/23_2_23/ZeroStyleCapPast'

        # suffix_name = src_dir_text_style.split('Cap')[-1] #39
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/4_3_23/res_f_36'
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/4_3_23/res_f_36'
        src_dir_text_style = '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_f_036/03_03_2023'
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_real_std'
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_f_036/03_03_2023'
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_f_036/03_03_2023'
        src_dir_text_style = os.path.join(os.path.expanduser('~'),'experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/23_03_2023')
        src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_mul_clip_style_v1_humor_test'
        src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_roberta_3_loss_v1_humor_test'
        src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_update_vit_style_v1_humor_test'
        # src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_update_vit_style_v1_romantic_test'
        # src_dir_text_style = os.path.join(base_path,'text_style')
        text_style_dir_path = os.listdir(src_dir_text_style)
        if factual_wo_prompt:
            tgt_path_text_style = os.path.join(src_dir_text_style,f'total_results_text_style_{suffix_name}_factual_wo_prompt.csv')
        else:
            tgt_path_text_style = os.path.join(src_dir_text_style,f'total_results_text_style_{suffix_name}.csv')
    else:
        src_dir_text_style = ''
        text_style_dir_path = ''
        tgt_path_text_style = ''

    # image_manipulation
    if 'image_manipulation' in exp_to_merge:
        # src_dir_image_manipulation = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/image_manipulation'
        src_dir_image_manipulation = os.path.join(base_path,'image_manipulation')
        image_manipulation_dir_path = os.listdir(src_dir_image_manipulation)
        if factual_wo_prompt:
            tgt_path_image_manipulation = os.path.join(src_dir_image_manipulation,'total_results_image_manipulation_factual_wo_prompt.csv')
        else:
            tgt_path_image_manipulation = os.path.join(src_dir_image_manipulation,'total_results_image_manipulation.csv')
    else:
        src_dir_image_manipulation = ''
        image_manipulation_dir_path = ''
        tgt_path_image_manipulation = ''

    debug_tgt_path_im_manipulation = os.path.join(os.path.expanduser('~'), 'results', cur_time+'_debug_total_results_image_manipulation.csv')
    debug_tgt_path_prompt_manipulation = os.path.join(os.path.expanduser('~'), 'results',
                                                cur_time+'_debug_total_results_prompt_manipulation.csv')

    src_dirs = {"prompt_manipulation":  src_dir_prompt_manipulation, "image_manipulation": src_dir_image_manipulation, "image_and_prompt_manipulation": src_dir_image_and_prompt_manipulation, "zerostylecap": src_dir_text_style}
    res_paths = {"prompt_manipulation": prompt_manipulation_dir_path, "image_manipulation": image_manipulation_dir_path, "image_and_prompt_manipulation": image_and_prompt_manipulation_dir_path, "zerostylecap":text_style_dir_path}
    tgt_paths = {"prompt_manipulation": tgt_path_prompt_manipulation,"image_manipulation": tgt_path_image_manipulation, "image_and_prompt_manipulation": tgt_path_image_and_prompt_manipulation, "zerostylecap": tgt_path_text_style}
    tgt_paths_debug = {"prompt_manipulation": debug_tgt_path_prompt_manipulation,"image_manipulation": debug_tgt_path_im_manipulation, "image_and_prompt_manipulation": "debug_tgt_path_image_and_prompt_manipulation"}

    return res_paths, src_dirs, tgt_paths
def get_all_paths(cur_time, factual_wo_prompt, exp_to_merge, suffix_name):
    # exp_to_merge = ["prompt_manipulation", "image_and_prompt_manipulation", "image_manipulation", "text_style"]
    # base_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_embed_debug_loss'
    # base_path = '/Users/danielabendavid/experiments/zero_style_cap/senticap/style_embed/senticap_StylizedZeroCap_my_embedding_model_8'
    base_path = '/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_audio_laughter_kids1_sw_f_zerocap/28_09_2023'

    #text_style
    if 'zerostylecap' in exp_to_merge:
        # src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/text_style'
        # 12.2.23
        # src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/12_2_23/text_style'
        # 20.2.23
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/20_2_23/ZeroStyleCap_8'
        # 23.2.23
        # src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/23_2_23/ZeroStyleCapPast'

        # suffix_name = src_dir_text_style.split('Cap')[-1] #39
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/4_3_23/res_f_36'
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/4_3_23/res_f_36'
        src_dir_text_style = '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_f_036/03_03_2023'
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_real_std'
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_f_036/03_03_2023'
        src_dir_text_style = os.path.join(os.path.expanduser('~'),'experiments/stylized_zero_cap_experiments/flickrstyle10k_ZeroStyleCap_embed/23_03_2023')
        # 1.4.23
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/erc_weighted_loss/28_04_2023/tmp'
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_StylizedZeroCap_roBERTa/val_set/03_05_2023'
        src_dir_text_style = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_StylizedZeroCap_roBERTa_val_set_weighted_loss/05_05_2023'
        # 14.5.23
        # src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/style_embed/senticap_StylizedZeroCap_my_embedding_model_8'
        # src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/style_embed/senticap_StylizedZeroCap_my_enbedding_model_real_std'
        # src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/erc/senticap_StylizedZeroCap_erc'
        src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/style_embed/senticap_StylizedZeroCap_stop_in_good_res/15_05_2023'
        # 22.5.23
        src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/baseline/image_manipulation'
        # 29.5.23
        src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/finetuned_roberta_best_sweep'
        #erc
        src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/erc/senticap_StylizedZeroCap_erc'
        # 31.5.23
        # src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/style_embed/senticap_StylizedZeroCap_my_embedding_model_8'
        # src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/finetuned_roberta_best_sweep'
        # 2.6.23
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/source_zero_stylecap/29_05_2023"
        # 4.6.23
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/style_embed/StylizedZeroCap_update_vit_along_iteration_global_prms/04_06_2023"
        # 5.6.23 - test 2
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/style_embed/StylizedZeroCap_update_vit_along_iteration_global_prms_test2/05_06_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v_test_neg"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v_test_pos"
        #test with update vit-pos
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v_test_pos"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v_test_neg"
        #3losses
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v101pos_test"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v101neg_test"
        # src_dir_text_style = '/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v18pos/16_06_2023/tmp'#check sweep 30 for updat vit
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v17pos/16_06_2023"
        #mul clip style neg best fluency
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_mul_clip_style_roberta_v20neg_test_best_fluency.yaml/19_06_2023"
        #final
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v14neg_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v14neg_test_best_fluence"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v15pos_test_best_fluence"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_mul_clip_style_wo_update_clip_v19pos_test_best_fluency"

        #3losses-best_fluency -pos
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v101pos_test_best_fluency/18_06_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v101pos_test/19_06_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v101pos_test/18_06_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v101pos_test_best_fluency"
        #mul clip-style - neg-best-fluency
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_mul_clip_style_roberta_v20neg_test_best_fluency.yaml/19_06_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v101pos_test/18_06_2023"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_mul_v1_neg_test"
        #3 losses -neg test
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_3_loss_v102neg_test/18_06_2023"
        #update vit pos after fix
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v101_pos_test/22_06_2023"
        # update vit pos after fix -same params - neg
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v14neg_test_best_fluence_fixed_same_params"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v100neg_test_best_fluence/22_06_2023"
        # update vit pos after fix -same params - pos
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v15pos_test_best_fluence_fixed_same_params"
        # flickrstyle10k
        #3 loss
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_roberta_3_loss_v1_romantic_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_roberta_3_loss_v1_humor_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_roberta_3_loss_v2_humor_test"

        #mul clip style
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_mul_clip_style_v1_romantic_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_mul_clip_style_v1_humor_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_mul_clip_style_v2_humor_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_mul_clip_style_v2_romantic_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_mul_v2_test/28_07_2023"

        #update vit
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_update_vit_style_v1_humor_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_update_vit_style_v1_romantic_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_update_vit_style_v2_humor_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_update_vit_style_v2_romantic_test"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v20neg/28_07_2023"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v30neg"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v30neg_v30"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v30neg_v31"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v32neg_test/31_07_2023"
        src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_audio_laughter_kids1_sw_f_zerocap/28_09_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/zerocap_im/28_09_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_audio_laughter_kids1_sw_f_zerocap_relevant/28_09_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_audio_cry_kids1_laughter_params/28_09_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v21neg/28_07_2023"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v22neg/28_07_2023"


        #Apollocap
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/Apollo_decent_pos_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/Apollo_decent_neg_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_Apollo_decent_humor_v1_test"
        # src_dir_text_style = "/Users/danielabendavid/experiments/zero_style_cap/flickrstyle10k/emoji/StylizedZeroCap_Apollo_decent_romantic_v1_test"



        suffix_name = src_dir_text_style.split('/')[-1]
        # src_dir_text_style = os.path.join(base_path,'text_style')
        # text_style_dir_path = os.listdir(src_dir_text_style)
        text_style_dir_path = src_dir_text_style
        if factual_wo_prompt:
            tgt_path_text_style = os.path.join(src_dir_text_style,f'total_results_text_style_{suffix_name}_factual_wo_prompt.csv')
        else:
            tgt_path_text_style = os.path.join(src_dir_text_style,f'total_results_text_style_{suffix_name}.csv')
    else:
        src_dir_text_style = ''
        text_style_dir_path = ''
        tgt_path_text_style = ''

    # prompt_manipulation
    if 'prompt_manipulation' in exp_to_merge:
        # src_dir_prompt_manipulation = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/prompt_manipulation'
        src_dir_prompt_manipulation = os.path.join(base_path,'prompt_manipulation')
        prompt_manipulation_dir_path = os.listdir(src_dir_prompt_manipulation)
        if factual_wo_prompt:
            tgt_path_prompt_manipulation = os.path.join(src_dir_prompt_manipulation,'total_results_prompt_manipulation_factual_wo_prompt.csv')
        else:
            tgt_path_prompt_manipulation = os.path.join(src_dir_prompt_manipulation,'total_results_prompt_manipulation.csv')
    else:
        src_dir_prompt_manipulation = ''
        prompt_manipulation_dir_path = ''
        tgt_path_prompt_manipulation = ''

    #image and prompt_manipulation
    if 'image_and_prompt_manipulation' in exp_to_merge:
        src_dir_image_and_prompt_manipulation = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/image_and_prompt_manipulation'
        src_dir_image_and_prompt_manipulation = os.path.join(base_path,'image_and_prompt_manipulation')
        image_and_prompt_manipulation_dir_path = os.listdir(src_dir_image_and_prompt_manipulation)
        if factual_wo_prompt:
            tgt_path_image_and_prompt_manipulation = os.path.join(src_dir_image_and_prompt_manipulation,'total_results_image_and_prompt_manipulation_factual_wo_prompt.csv')
        else:
            tgt_path_image_and_prompt_manipulation = os.path.join(src_dir_image_and_prompt_manipulation,'total_results_image_and_prompt_manipulation.csv')
    else:
        src_dir_image_and_prompt_manipulation = ''
        image_and_prompt_manipulation_dir_path = ''
        tgt_path_image_and_prompt_manipulation = ''


    # image_manipulation
    if 'image_manipulation' in exp_to_merge:
        # src_dir_image_manipulation = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/7_2_23/image_manipulation'
        src_dir_image_manipulation = os.path.join(base_path,'image_manipulation')
        image_manipulation_dir_path = os.listdir(src_dir_image_manipulation)
        if factual_wo_prompt:
            tgt_path_image_manipulation = os.path.join(src_dir_image_manipulation,'total_results_image_manipulation_factual_wo_prompt.csv')
        else:
            tgt_path_image_manipulation = os.path.join(src_dir_image_manipulation,'total_results_image_manipulation.csv')
    else:
        src_dir_image_manipulation = ''
        image_manipulation_dir_path = ''
        tgt_path_image_manipulation = ''

    debug_tgt_path_im_manipulation = os.path.join(os.path.expanduser('~'), 'results', cur_time+'_debug_total_results_image_manipulation.csv')
    debug_tgt_path_prompt_manipulation = os.path.join(os.path.expanduser('~'), 'results',
                                                cur_time+'_debug_total_results_prompt_manipulation.csv')

    src_dirs = {"prompt_manipulation":  src_dir_prompt_manipulation, "image_manipulation": src_dir_image_manipulation, "image_and_prompt_manipulation": src_dir_image_and_prompt_manipulation, "zerostylecap": src_dir_text_style}
    res_paths = {"prompt_manipulation": prompt_manipulation_dir_path, "image_manipulation": image_manipulation_dir_path, "image_and_prompt_manipulation": image_and_prompt_manipulation_dir_path, "zerostylecap":text_style_dir_path}
    tgt_paths = {"prompt_manipulation": tgt_path_prompt_manipulation,"image_manipulation": tgt_path_image_manipulation, "image_and_prompt_manipulation": tgt_path_image_and_prompt_manipulation, "zerostylecap": tgt_path_text_style}
    tgt_paths_debug = {"prompt_manipulation": debug_tgt_path_prompt_manipulation,"image_manipulation": debug_tgt_path_im_manipulation, "image_and_prompt_manipulation": "debug_tgt_path_image_and_prompt_manipulation"}

    return res_paths, src_dirs, tgt_paths

def get_missed_img_nums(f1,f2):
    '''
    f1- good file
    f2 - missed file
    '''
    def get_first_col(f1):
        c = []
        with open(f1, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i,row in enumerate(reader):
                if i==0:
                    continue
                c.append(int(row[0]))
        return c
    c1 = get_first_col(f1)
    c2 = get_first_col(f2)
    diff = list(set(c1)-set(c2))
    print(f"The difference is: {diff}")
    missed_img_nums = diff
    return missed_img_nums


def get_list_of_files():
    list_of_files = ['/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/17_03_2023/tmp/results_10_55_24__17_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/17_03_2023/tmp/results_10_55_35__17_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/17_03_2023/tmp/results_10_55_57__17_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/17_03_2023/tmp/results_10_56_36__17_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/17_03_2023/tmp/results_10_56_41__17_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/17_03_2023/tmp/results_10_56_52__17_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/17_03_2023/tmp/results_10_56_59__17_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/18_03_2023/tmp/results_21_55_10__18_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/18_03_2023/tmp/results_21_55_15__18_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/18_03_2023/tmp/results_21_55_26__18_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/18_03_2023/tmp/results_21_56_02__18_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/19_03_2023/tmp/results_09_36_22__19_03_2023.csv',
                     '/home/nlp/tzufar/experiments/stylized_zero_cap_experiments/senticap_ZeroStyleCap_with_emoji/19_03_2023/tmp/results_11_40_14__19_03_2023.csv']
    return list_of_files

def map_img_name_to_idx(dataset,test_split):
    img_name_to_idx = {}
    for i, im in enumerate(os.listdir(
        os.path.join(os.path.join(os.path.expanduser('~'), 'data', dataset, 'images',test_split)))):
        if ('.jpg' or '.jpeg' or '.png') not in im:
            continue
        if dataset=='senticap':
            img_name = int(im.rsplit('.')[0])
        elif dataset == 'flickrstyle10k':
            img_name = im.rsplit('.')[0]
        img_name_to_idx[img_name] = i
        if i == 119:
            print(f"i={i},img_name={img_name}")
    return img_name_to_idx

# img_nums=[]
# for i in img_name_to_idx:
#     if img_name_to_idx[i] in [119,126,559]:
#         print(f"i={i},img_name_to_idx={img_name_to_idx[i]}")
#         img_nums.append(i)
# print(img_nums)


def get_img_idx_to_name(caption_img_dict):
    imgs_to_test = []
    for setdir in aption_img_dict:
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

# [225571, 471814, 72873, 357322, 106314, 368459, 575135, 423830, 51258, 265596, 551518, 448703]
def main():
    # missed_img_nums = get_missed_img_nums(f1, f2)
    args = get_args()
    cuda_idx = args.cuda_idx_num
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print(f'Cur time is: {cur_time}')
    # get_img_idx_to_name(args.caption_img_dict)
    factual_wo_prompt = False
    use_factual = False
    t = {"prompt_manipulation": "img_num","image_manipulation": "img_num\style",  "image_and_prompt_manipulation": "img_num\style", "text_style": "img_num"}

    exp_to_merge = ["zerostylecap"]
    # exp_to_merge = ["prompt_manipulation", "image_and_prompt_manipulation", "image_manipulation", "zerostylecap"]

    # dataset = 'flickrstyle10k' #'senticap'
    dataset = 'senticap' #'senticap'
    test_split = 'test' #'test'
    merge_dirs = False
    merge_res_of_sweep = False

    if merge_dirs:
        suffix_name = ''
        res_paths, src_dirs, tgt_paths = get_all_paths(cur_time, factual_wo_prompt, exp_to_merge, suffix_name)  # todo:
        img_name_to_idx = map_img_name_to_idx(dataset, test_split)
        if merge_res_of_sweep:
            merge_res_sweep_to_one(exp_to_merge, res_paths, src_dirs, t, tgt_paths, factual_wo_prompt, use_factual,
                               img_name_to_idx)
        else:
            merge_res_files_to_one(exp_to_merge, res_paths, src_dirs, t, tgt_paths, factual_wo_prompt, use_factual,
                                   img_name_to_idx)
    else: # for the case of file list
        # file_list = get_list_of_files()
        # dir_files = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/erc_weighted_loss/28_04_2023/tmp'
        dir_files = '/Users/danielabendavid/results/zero_style_cap/erc_old_params'
        dir_files = '/Users/danielabendavid/experiments/zero_style_cap/senticap/baseline/prompt_manipulation'
        dir_files = '/Users/danielabendavid/experiments/zero_style_cap/senticap/source_zero_stylecap/29_05_2023'
        dir_files = '/Users/danielabendavid/experiments/zero_style_cap/senticap/baseline/prompt_manipulation'
        dir_files = '/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_update_vit_focus_clip_v18pos/16_06_2023/tmp'
        dir_files = '/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_roberta_mul_v1_neg_test/26_07_2023/tmp'
        dir_files = '/Users/danielabendavid/experiments/zero_style_cap/senticap/audio/StylizedZeroCap_audio_crying_kids1/23_09_2023/tmp'
        dir_files = '/Users/danielabendavid/experiments/zero_style_cap/senticap/audio/StylizedZeroCap_audio_laughter_kids1/23_09_2023/tmp'
        dir_files = '/Users/danielabendavid/experiments/zero_style_cap/senticap/roberta/StylizedZeroCap_audio_ambulance3_test/30_01_2024/tmp/tmp'

        file_list = [os.path.join(dir_files,f) for f in os.listdir(dir_files) if f.endswith('.csv') and f.startswith('results')]
        tgt_path = os.path.join(dir_files,'total_results_text_style_tmp.csv')
        merge_list_res_files_to_one(file_list, tgt_path)
    print("finish program")


if __name__ == "__main__":
    main()
