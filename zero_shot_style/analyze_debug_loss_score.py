import csv
import os
import pandas as pd


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
def extract_loss_data(debug_file_path):
    with open(debug_file_path,'r') as fp:
        lines = fp.readlines()
    all_data = {}
    single_data = {}
    iter_num = -1

    CLIPScore
    style_classification
    fluency
    for i,line in enumerate(lines):
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
            if single_data["img_idx"] not in all_data:
                all_data[single_data["img_idx"]] = {single_data["style"]: single_data}
            else:
                all_data[single_data["img_idx"]][single_data["style"]] =  single_data
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

def add_eval_data(all_data,eval_debug_file_path, desired_scores):
    data = pd.read_csv(eval_debug_file_path)
    for i in range(len(data)):
        if data.iloc[i, get_score_type_idx(data.columns, 'Dataset')] == 'ZeroStyleCap':
            style = data.iloc[i, get_score_type_idx(data.columns,'style')]
            for score_type in desired_scores:
                all_data[str(i)][style][score_type] = data.iloc[i, get_score_type_idx(data.columns, score_type)]
    return all_data

def main():
    debug_file_path = '/Users/danielabendavid/projects/zero-shot-style/zero_shot_style/debug/debug_loss_flickr_23_2_v0.txt'
    eval_debug_file_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/evaluation_all_frames_all_models_24_2_23.csv'
    desired_scores = ['CLIPScore', 'style_classification', 'fluency']
    log_dir,log_file_name = debug_file_path.rsplit('/',1)
    tgt_debug_results_path = os.path.join(log_dir,'analyzed_log_'+log_file_name.split('.txt')[0]+'.csv')
    all_data = extract_loss_data(debug_file_path)
    all_data = add_eval_data(all_data,eval_debug_file_path, desired_scores)
    write_results(all_data,tgt_debug_results_path)
    print("finish main")


if __name__=='__main__':
    main()