import csv
import os
import pickle
import statistics
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

import numpy as np
import pandas as pd
import yaml

from zero_shot_style.mapping_idx_to_img_name import get_mapping_idx_img_name


def write_results(all_data, tgt_debug_results_path):
    print(f'Writing results into: {tgt_debug_results_path}')
    title = ['img_idx', 'img_name', 'style', 'word_num', 'iter_num','fluency_score', 'clip_score', 'style_score', 'ce_loss', 'clip_loss', 'text_style_loss','ce_loss_w_scale', 'clip_loss_w_scale', 'text_style_losses_w_scale', 'ce_losses', 'clip_losses', 'text_style_losses', 'best_captions_up_to_now', 'total_best_caption']
    with open(tgt_debug_results_path, 'w') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(title)
        for img_idx in all_data:
            row = []
            for t in ['img_idx', 'img_name', 'style', 'word_num']:
                row.append(all_data[img_idx][t])
            for word_num in list(all_data[img_idx]['best_captions_up_to_now'].keys()):
                row.append(word_num)
                for iter_num in range(len(all_data[img_idx]['ce_loss'])):
                    row.append(iter_num)
                    for score in ['fluency_score', 'clip_score', 'style_score']:
                        row.append(all_data[img_idx][score])
                    for l in ['ce_loss', 'clip_loss', 'text_style_loss','ce_loss_w_scale', 'clip_loss_w_scale', 'text_style_losses_w_scale', 'ce_losses', 'clip_losses', 'text_style_losses']:
                        row.append(all_data[img_idx][l][word_num][iter_num])
                row.append(all_data[img_idx]['best_captions_up_to_now'][word_num])
            row.append(all_data[img_idx]['total_best_caption'])


def update_val(single_data, category, word_num, iter_num, line):
    if category not in single_data:
        single_data[category] = {word_num: {}}
    elif word_num not in single_data[category]:
            single_data[category][word_num] = {}
    single_data[category][word_num][iter_num] = line.split(' = ')[1].split('\n')[0]


def extract_loss_data(debug_file_paths, relevant_img_names):
    all_data = {}
    for test_type in debug_file_paths:
        if test_type not in all_data:
            all_data[test_type] = {}
        with open(debug_file_paths[test_type],'r') as fp:
            lines = fp.readlines()
        single_data = {}
        iter_num = -1
        # CLIPScore
        # style_classification
        # fluency
        for i,line in enumerate(lines):
            if len(all_data)>20:
                break#todo: remove it. it is just for debug
            # clip loss
            if "clip_loss = " in line:
                update_val(single_data, "clip_loss", word_num, iter_num, line)
            elif "clip_losses = " in line:
                update_val(single_data, "clip_losses", word_num, iter_num, line)
            elif "clip_loss with scale " in line:
                update_val(single_data, "clip_loss_w_scale", word_num, iter_num, line)
            # ce loss
            elif "ce_loss = " in line:
                update_val(single_data, "ce_loss", word_num, iter_num, line)
            elif "ce_losses = " in line:
                update_val(single_data, "ce_losses", word_num, iter_num, line)
            elif "ce_loss with scale " in line:
                update_val(single_data, "ce_loss_w_scale", word_num, iter_num, line)
            # style loss
            elif "text_style_loss = " in line:
                update_val(single_data, "text_style_loss", word_num, iter_num, line)
            elif "text_style_losses = " in line:
                update_val(single_data, "text_style_losses", word_num, iter_num, line)
            elif "text_style_loss with scale = " in line:
                update_val(single_data, "text_style_losses_w_scale", word_num, iter_num, line)

            # elif "after calc clip loss" in line: # start new iteration
            #     iter_num += 1
            elif " iter num = " in line: # start new iteration
                iter_num = int(line.split("iter num = ")[1][0])
            elif "~~~~~~" in line: # start new caption
                word_num = 0
            elif "| Work on img path:" in line: # start new image caption
                single_data["img_name"] = line.split('/')[-1].split('.jpg')[0]
            elif "style of: ***" in line:
                single_data["style"] = line.split("*** ")[1].split(' ')[0]
            #final single  data
            elif "best clip: " in line:
                single_data["best_caption"] = line.split("best clip: ")[1][:-1]
                if int(single_data["img_name"]) in relevant_img_names:
                    if single_data["img_idx"] not in all_data[test_type]:
                        all_data[test_type][single_data["img_idx"]] = {single_data["style"]: single_data}
                    else:
                        all_data[test_type][single_data["img_idx"]][single_data["style"]] = single_data
                single_data = {}
            elif line[:len("Img num = ")]=="Img num = ":
                single_data["img_idx"] = line.split(' ')[-1][:-1]

            #best_caption for word_num
            elif " | " in line:
                if "'best_captions_up_to_now'" not in single_data:
                    single_data["'best_captions_up_to_now'"] = {}
                single_data["'best_captions_up_to_now'"][word_num] = line.split(" | ")[1][:-1]
                word_num += 1
                iter_num = -1
    return all_data


def get_score_type_idx(data_columns,score_type):
    for i in range(len(data_columns)):
        if data_columns[i] == score_type:
            return i


def replace_user_home_dir(path):
    if str(path)[0] == '~':
        path = os.path.join(os.path.expanduser('~'), path[2:])
    elif str(path).split('/')[1] == 'Users':
        path = os.path.join(os.path.expanduser('~'), "/".join(path.split('/')[3:]))
    elif '/' in str(path) and str(path).split('/')[1] == 'home':
        if str(path).split('/')[2] == 'bdaniela':
            path = os.path.join(os.path.expanduser('~'), "/".join(path.split('/')[3:]))
        else:
            path = os.path.join(os.path.expanduser('~'), "/".join(path.split('/')[4:]))
    return path



def get_scores_data(score_files, mapping_img_name2idx, desired_scores):
    all_data = {}
    print("start to get score data")
    for exp in score_files:
        if exp not in all_data:
            all_data[exp] = {}
        eval_data = pd.read_csv(score_files[exp])
        for i in range(len(eval_data)):
            test_name = eval_data.iloc[i, get_score_type_idx(eval_data.columns, 'test_name')]
            if test_name.startswith('capdec'):
                test_name = 'capdec'
            style = eval_data.iloc[i, get_score_type_idx(eval_data.columns,'style')]
            img_name = eval_data.iloc[i, get_score_type_idx(eval_data.columns, 'k')]
            img_idx = str(mapping_img_name2idx[img_name])
            image_path = replace_user_home_dir(eval_data.iloc[i, get_score_type_idx(eval_data.columns, 'image_path')])
            if img_idx not in all_data[exp]:
                all_data[exp][img_idx] = {}
            if style not in all_data[exp][img_idx]:
                all_data[exp][img_idx][style] = {}
            all_data[exp][img_idx][style]['img_name'] = img_name
            all_data[exp][img_idx][style]['img_path'] = image_path
            all_data[exp][img_idx][style]['caption'] = eval_data.iloc[i, get_score_type_idx(eval_data.columns, 'res')]
            all_data[exp][img_idx][style]['img_idx'] = img_idx
            for score_type in desired_scores:
                all_data[test_name][img_idx][style][score_type] = eval_data.iloc[i, get_score_type_idx(eval_data.columns, score_type)]
    print("finish to get score data")
    return all_data


def add_eval_data(all_data,eval_debug_file_path, desired_scores, mapping_img_name2idx):
    eval_data = pd.read_csv(eval_debug_file_path)
    for i in range(len(eval_data)):
        test_name = eval_data.iloc[i, get_score_type_idx(eval_data.columns, 'test_name')]
        if test_name == 'ZeroStyleCap':
            style = eval_data.iloc[i, get_score_type_idx(eval_data.columns,'style')]
            img_idx = str(mapping_img_name2idx[eval_data.iloc[i, get_score_type_idx(eval_data.columns, 'k')]])
            if img_idx in all_data[test_name] and style in all_data[test_name][img_idx]:
                for score_type in desired_scores:
                    all_data[test_name][img_idx][style][score_type] = eval_data.iloc[i, get_score_type_idx(eval_data.columns, score_type)]
    return all_data

def plot_statistic_data(final_statistic_mean, final_statistic_std,title):
    fig, axs = plt.subplots(len(final_statistic_mean))
    fig.suptitle(title)
    for i,loss_i in enumerate(final_statistic_mean):
        word_num = np.arange(0, len(final_statistic_mean[loss_i]))
        axs[i].errorbar(word_num, final_statistic_mean[loss_i], final_statistic_std[loss_i], linestyle='None', marker='^')
        axs[i].set_title(loss_i)
    fig.tight_layout()
    plt.savefig(os.path.join(os.path.expanduser('~'),"results/zero_style_cap/weighted_loss",title+"_loss_val_statistic.png"))
    plt.show(block=False)

def plot_final_statistic_data(final_statistic_mean_all_data, final_statistic_std_all_data, final_statistic_mean_best_data, final_statistic_std_best_data, title):
    fig, axs = plt.subplots(len(final_statistic_mean_all_data))
    fig.suptitle(title)
    for i,loss_i in enumerate(final_statistic_mean_all_data):
        word_num = np.arange(0, len(final_statistic_mean_all_data[loss_i]))
        axs[i].errorbar(word_num, final_statistic_mean_all_data[loss_i], final_statistic_std_all_data[loss_i], linestyle='None', marker='^', label='all_data')
        axs[i].errorbar(word_num, final_statistic_mean_best_data[loss_i], final_statistic_std_best_data[loss_i], linestyle='None', marker='^', label='best_data')
        axs[i].set_title(loss_i)
        axs[i].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    fig.tight_layout()
    plt.savefig(os.path.join(os.path.expanduser('~'),"results/zero_style_cap/weighted_loss",title+"_loss_val_statistic.png"))
    plt.show(block=False)


def get_loss_data(all_data,final_iterations_idx, loss_data_path):
    clip_loss_all_words = {}
    ce_loss_all_words = {}
    style_loss_all_words = {}
    loss_all_data = {'clip_loss': clip_loss_all_words, 'ce_loss': ce_loss_all_words, 'text_style_loss': style_loss_all_words}
    desired_losses = ['clip_loss', 'ce_loss', 'text_style_loss']
    for test_name in all_data:
        for img_idx in all_data[test_name]:
            for style in all_data[test_name][img_idx]:
                for loss_i in desired_losses:
                    for word_i in all_data[test_name][img_idx][style][loss_i]:
                        if word_i not in loss_all_data[loss_i]:
                            loss_all_data[loss_i][word_i] = []
                        if not np.isnan(float(all_data[test_name][img_idx][style][loss_i][word_i][
                                                  final_iterations_idx])):  # todo:check why there re None values
                            loss_all_data[loss_i][word_i].append(
                                float(all_data[test_name][img_idx][style][loss_i][word_i][final_iterations_idx]))
                        # if np.isnan(float(all_data[test_name][img_idx][style][loss_i][word_i][final_iterations_idx])):
                        #     print("check:")
                        #     print(f"test_name: {test_name}, img_idx: {img_idx}, style: {style}, loss_i: {loss_i}, word_i: {word_i}, final_iterations_idx: {final_iterations_idx}")
                        # else:
                        #     if not np.isnan(float(all_data[test_name][img_idx][style][loss_i][word_i][final_iterations_idx])): #todo:check why there re None values
                        #         loss_all_data[loss_i][word_i].append(float(all_data[test_name][img_idx][style][loss_i][word_i][final_iterations_idx]))
    with open(loss_data_path, "wb") as f:
        pickle.dump(loss_all_data, f)
    # with open(tgt_path, 'rb') as f:
    #     restored_statistic_loss_all_words = pickle.load(f)
    return loss_all_data


def get_statiistic_data(all_data,final_iterations_idx, loss_data_path):
    '''
    :param all_data:
    :return:
    '''
    loss_all_data = get_loss_data(all_data,final_iterations_idx, loss_data_path)
    final_statistic_mean = {}
    final_statistic_std = {}
    for i,loss_i in enumerate(loss_all_data):
        final_statistic_mean[loss_i] = []
        final_statistic_std[loss_i] = []
        for word_i in loss_all_data[loss_i]:
            final_statistic_mean[loss_i].append(statistics.mean(loss_all_data[loss_i][word_i]))
            final_statistic_std[loss_i].append(statistics.stdev(loss_all_data[loss_i][word_i]))
    return final_statistic_mean, final_statistic_std


def write_to_pdf(all_data, desired_scores, tgt_pdf_file_path):
    print(f"writing results into {tgt_pdf_file_path}")
    # create a canvas
    c = canvas.Canvas(tgt_pdf_file_path, pagesize=letter)
    # set the font size
    c.setFont("Helvetica", 10)
    img_width = 130
    img_height = 130
    y = 0
    for i, idx in enumerate(all_data["ZeroStyleCap"]):
        # if i<5: #todo
        #     continue

        y = y + 30
        best_captions = {}
        for ei,exp in enumerate(all_data):
            if ei>0:
                y = y + 20
            for style in all_data[exp][idx]:
                image_num = int(all_data[exp][idx][style]['img_name'])
                image_idx = all_data[exp][idx][style]['img_idx']
                image_path = all_data[exp][idx][style]['img_path']
                best_captions[style] = all_data[exp][idx][style]['caption']

            #write stylized captions
            for s,style in enumerate(best_captions):
                # write scores of stylized captions
                y = y + 20 * s
                x = 20
                for si,score_type in enumerate(desired_scores):
                    if score_type in all_data[exp][idx][style]:
                        c.drawString(x, y, f"{score_type}: {all_data[exp][idx][style][score_type]},")
                        score_len = c.stringWidth(f"{score_type}: {all_data[exp][idx][style][score_type]},")
                        x = x+score_len+1
                y = y + 15
                # write stylized captions
                max_length_of_line = 102
                text2write=f"{exp},{style}: {best_captions[style]}"
                if len(text2write)>max_length_of_line:
                    final_idx = text2write[:max_length_of_line].rfind(' ')
                    part_a = text2write[:max_length_of_line][:final_idx]
                    part_b = text2write[:max_length_of_line][final_idx: ] + text2write[max_length_of_line:]
                    c.drawString(20, y, part_b)
                    y = y + 15
                    c.drawString(20, y, part_a)
                else:
                    c.drawString(20, y, f"{exp},{style}: {best_captions[style]}")

        title = f"image_num={image_num}, idx={image_idx}"

        #write image
        y = y+20
        c.drawImage(image_path, 20,
                    y,
                    width=img_width, height=img_height)
        #write title
        y = y + img_height + 15
        c.drawString(20, y, title)
        y = y + 15
        c.drawString(20, y, "----------------------------------------------------------------------")

        if (i+1) % 2 == 0 and i != 0:
            c.showPage()
            y = 0
            # break#todo
    c.showPage()  # add this line to indicate the end of the last page
    c.save()

    print(f"finish to save pdf of the results into: {tgt_pdf_file_path}")

def get_best_data(all_data, desired_scores, scores_th):
    '''
    :param all data:{test_type:{img_idx:{style:{img_idx:,img_name:,style:,clip_loss:{word_num:{beam_num:}}}}}}
    :return:
    '''
    # get best dsta according to threshold scores in scores_th
    best_data = {"ZeroStyleCap":{}}
    for idx in all_data["ZeroStyleCap"]:
        for style in all_data["ZeroStyleCap"][idx]:
            good_score = True
            for score_type in desired_scores:
                if score_type in all_data["ZeroStyleCap"][idx][style]:
                    if all_data["ZeroStyleCap"][idx][style][score_type] < scores_th[score_type]:
                        good_score = False
            if score_type in all_data["ZeroStyleCap"][idx][style] and good_score:
                if idx not in best_data["ZeroStyleCap"]:
                    best_data["ZeroStyleCap"][idx] = {}
                best_data["ZeroStyleCap"][idx][style] = all_data["ZeroStyleCap"][idx][style]
    return best_data

def merge_debug_files(debug_dir_for_file_paths, merged_debug_file_name):
    debug_file_paths = {}
    for test_name in debug_dir_for_file_paths:
        merged_lines = []
        for f in os.listdir(debug_dir_for_file_paths[test_name]):
            if f.endswith('.txt'):
                with open(os.path.join(debug_dir_for_file_paths[test_name],f),'r') as fp:
                    lines = fp.readlines()
                merged_lines.extend(lines)
        with open(os.path.join(debug_dir_for_file_paths[test_name],merged_debug_file_name),'w') as fp:
            fp.writelines(merged_lines)
        debug_file_paths[test_name] = os.path.join(debug_dir_for_file_paths[test_name],merged_debug_file_name)
        print(f"finished to merge debug files into {os.path.join(debug_dir_for_file_paths[test_name],merged_debug_file_name)}")
        return debug_file_paths


def main():
    configfile = os.path.join('.', 'configs', 'config.yaml')
    score_files = {"ZeroStyleCap": "/Users/danielabendavid/experiments/zero_style_cap/senticap/style_embed/senticap_StylizedZeroCap_my_embedding_model_real_std/evaluation_all_frames_real_std.csv",
                   "capdec": "/Users/danielabendavid/experiments/capdec/27_2_23/evaluation_all_frames_capdec.csv"}
    data_split = 'test' # 'test'
    base_dir4tgt_pdf_file_path = os.path.join(os.path.expanduser('~'),'experiments')
    img_dir_path = os.path.join(os.path.expanduser('~'),'data/senticap/images/'+data_split)
    desired_scores = ['fluency', 'CLIPScore', 'style_classification']
    scores_th = {'fluency':0.9, 'CLIPScore':0.3, 'style_classification':1}
    # scores_th = {'fluency':0.9, 'CLIPScore':0.32, 'style_classification':1}
    all_model_names = '_'.join([k for k in score_files])
    all_data_tgt_pdf_file_name = f"res_all_data_{all_model_names}.pdf"
    tgt_pdf_file_path = os.path.join(base_dir4tgt_pdf_file_path, all_data_tgt_pdf_file_name)

    mapping_idx2img_name, mapping_img_name2idx = get_mapping_idx_img_name(configfile)
    # scores data
    all_data = get_scores_data(score_files, mapping_img_name2idx, desired_scores)
    write_to_pdf(all_data, desired_scores, tgt_pdf_file_path)

    print("finish main")


if __name__=='__main__':
    main()