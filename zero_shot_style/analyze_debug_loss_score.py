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


def extract_loss_data(debug_file_paths):
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

            elif "after calc clip loss" in line: # start new iteration
                iter_num += 1
            elif "~~~~~~" in line: # start new caption
                word_num = 0
            elif "| Work on img path:" in line: # start new image caption
                single_data["img_name"] = line.split('/')[-1].split('.jpg')[0]
            elif "style of: ***" in line:
                single_data["style"] = line.split("*** ")[1].split(' ')[0]
            elif "best clip: " in line:
                single_data["best_caption"] = line.split("best clip: ")[1][:-1]
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


def add_eval_data(all_data,eval_debug_file_path, desired_scores, mapping_img_name2idx):
    eval_data = pd.read_csv(eval_debug_file_path)
    for i in range(len(eval_data)):
        test_name = eval_data.iloc[i, get_score_type_idx(eval_data.columns, 'test_name')]
        if test_name == 'ZeroStyleCap':
            style = eval_data.iloc[i, get_score_type_idx(eval_data.columns,'style')]
            for score_type in desired_scores:
                all_data[test_name][str(mapping_img_name2idx[eval_data.iloc[i, get_score_type_idx(eval_data.columns,'k')]])][style][score_type] = eval_data.iloc[i, get_score_type_idx(eval_data.columns, score_type)]
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

def write_to_pdf(all_data, img_dir_path, desired_scores, tgt_pdf_file_path):
    # create a canvas
    c = canvas.Canvas(tgt_pdf_file_path, pagesize=letter)
    # set the font size
    c.setFont("Helvetica", 10)
    img_width = 130
    img_height = 130
    y = 0
    for i, idx in enumerate(all_data["ZeroStyleCap"]):
        y = y + 30
        best_captions = {}
        for style in all_data["ZeroStyleCap"][idx]:
            image_num = int(all_data["ZeroStyleCap"][idx][style]['img_name'])
            image_idx = all_data["ZeroStyleCap"][idx][style]['img_idx']
            image_path = os.path.join(img_dir_path,all_data["ZeroStyleCap"][idx][style]['img_name']+'.jpg')
            best_captions[style] = all_data["ZeroStyleCap"][idx][style]['best_caption']
        title = f"image_num={image_num}, idx={image_idx}"

        #write stylized captions
        for s,style in enumerate(best_captions):
            # write scores of stylized captions
            y = y + 20*s
            x = 20
            for si,score_type in enumerate(desired_scores):
                if score_type in all_data["ZeroStyleCap"][idx][style]:
                    c.drawString(x, y, f"{score_type}: {all_data['ZeroStyleCap'][idx][style][score_type]},")
                    score_len = c.stringWidth(f"{score_type}: {all_data['ZeroStyleCap'][idx][style][score_type]},")
                    x = x+score_len+1
            y = y + 15
            # write stylized captions
            c.drawString(20, y, f"{style}: {best_captions[style]}")

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

        if (i+1) % 3 == 0 and i != 0:
            c.showPage()
            y = 0
    c.showPage()  # add this line to indicate the end of the last page
    c.save()
    print("finish to save pdf")

def get_best_data(all_data, desired_scores, scores_th):
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

def main():
    '''

    :param all data:{test_type:{img_idx:{style:{img_idx:,img_name:,style:,clip_loss:{word_num:{beam_num:}}}}}}
    :return:
    '''
    configfile = os.path.join('.', 'configs', 'config.yaml')
    debug_file_paths = {
        # "image_and_prompt_manipulation": "/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_fixed_param_25_3_23/image_and_prompt_manipulation/senticap_image_and_prompt_manipulation_debug.txt",
        # "image_manipulation": "/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_fixed_param_25_3_23/image_manipulation/senticap_image_manipulation_debug.txt",
        # "prompt_manipulation": "/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_fixed_param_25_3_23/prompt_manipulation/senticap_prompt_manipulation_debug.txt",
        "ZeroStyleCap": "/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_fixed_param_25_3_23/ZeroStyleCap/debug_loss_real_senticap_23_2_v0.txt"}
    eval_debug_file_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/senticap_fixed_param_25_3_23/evaluation_all_frames.csv'
    img_dir_path = os.path.join(os.path.expanduser('~'),'data/senticap/images/test')
    ZeroStyleCap_loss_data_path = os.path.join(os.path.expanduser('~'),
                                               "results/zero_style_cap/weighted_loss", "ZeroStyleCap_loss_data.pkl")
    desired_scores = ['fluency', 'CLIPScore', 'style_classification']
    # scores_th = {'fluency':0.9, 'CLIPScore':0.26, 'style_classification':1}
    scores_th = {'fluency':0.9, 'CLIPScore':0.32, 'style_classification':1}
    tgt_pdf_file_name = f"debug_res_fluency={scores_th['fluency']}_CLIPScore={scores_th['CLIPScore']}_style_classification={scores_th['style_classification']}.pdf"
    tgt_pdf_file_path = os.path.join(os.path.expanduser('~'),
                 "results/zero_style_cap/weighted_loss", tgt_pdf_file_name)
    loss_all_data_path = {"all_data": os.path.join(os.path.expanduser('~'), "results/zero_style_cap/weighted_loss",
                                  "loss_al_data.pkl"),
                          "best_data": os.path.join(os.path.expanduser('~'), "results/zero_style_cap/weighted_loss",
                                      "loss_al_data.pkl")}

    calc_from_scratch = False #
    final_iterations_idx = 4 #suppose there are only 5 iterations in the results,  so we take as statistic the loss res of iter idx 4

    eval_dir,eval_file_name = eval_debug_file_path.rsplit('/',1)
    tgt_debug_results_path = os.path.join(eval_dir,'analyzed_log.csv')

    if calc_from_scratch:
        mapping_idx2img_name, mapping_img_name2idx = get_mapping_idx_img_name(configfile)
        # loss data
        all_data = extract_loss_data(debug_file_paths)
        # scores data
        all_data = add_eval_data(all_data,eval_debug_file_path, desired_scores, mapping_img_name2idx)
        with open(ZeroStyleCap_loss_data_path, "wb") as f:
            pickle.dump(all_data, f)
    else:
        with open(ZeroStyleCap_loss_data_path, 'rb') as f:
            all_data = pickle.load(f)

    best_data = get_best_data(all_data, desired_scores, scores_th)


    write_to_pdf(best_data, img_dir_path, desired_scores, tgt_pdf_file_path)
    # group loss data only for successful examples
    final_statistic_mean_all_data, final_statistic_std_all_data = get_statiistic_data(all_data, final_iterations_idx,loss_all_data_path['all_data'])
    final_statistic_mean_best_data, final_statistic_std_best_data = get_statiistic_data(best_data, final_iterations_idx,loss_all_data_path['best_data'])
    plot_statistic_data(final_statistic_mean_all_data, final_statistic_std_all_data,'all_data')
    plot_statistic_data(final_statistic_mean_best_data, final_statistic_std_best_data,'best_data')
    plot_final_statistic_data(final_statistic_mean_all_data, final_statistic_std_all_data, final_statistic_mean_best_data, final_statistic_std_best_data, 'final')
    write_results(all_data,tgt_debug_results_path)
    print("statistics:")
    print("mean_clip_loss:")
    print(f"all data: {final_statistic_mean_all_data['clip_loss']}")
    print(f"best data: {final_statistic_mean_best_data['clip_loss']}")
    print("*********************")
    print("mean_ce_loss:")
    print(f"all data: {final_statistic_mean_all_data['ce_loss']}")
    print(f"best data: {final_statistic_mean_best_data['ce_loss']}")
    print("*********************")
    print("mean_text_style_loss:")
    print(f"all data: {final_statistic_mean_all_data['text_style_loss']}")
    print(f"best data: {final_statistic_mean_best_data['text_style_loss']}")
    print("*********************")

    print("finish main")


if __name__=='__main__':
    main()