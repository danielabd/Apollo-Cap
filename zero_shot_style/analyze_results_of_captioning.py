import os
from datetime import datetime

import pandas as pd

def get_set_params(section):
    for line in section:
        if imgs_dir in line:
            img_num = line.split(imgs_dir)[1].split(' with')[0]
        elif "style_type" in line:
            model = line.split("*** ")[1].split(" ***")[0]
        elif "style of" in line:
            label = line.split("*** ")[1].split(" ***")[0]
        elif "text_style_scale" in line:
            text_style_scale = line.split("*** ")[1].split(" ***")[0]
        elif "style_type" in line:
            model = line.split("*** ")[1].split(" ***")[0]
    return img_num,model,label,text_style_scale

def writing_results_to_csv(reults,tgt_path_results):
    print("Writing results to: "+tgt_path_results)
    pd_reults = pd.DataFrame(reults)
    pd_reults.to_csv(tgt_path_results)

def main():
    # base_dir = '/home/bdaniela/zero-shot-style/zero_shot_style'
    base_dir = '/home/bdaniela/zero-shot-style/results/00_27_27__21_07_2022'
    log_file_name = "00_27_27__21_07_2022_log.txt"
    log_file = os.path.join(base_dir,log_file_name)
    cur_time = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    print(f'Cur time is: {cur_time}')
    tgt_path_results = os.path.join(base_dir, f"{cur_time}_analyzed_results.csv")
    start_of_new_section = False
    end_of_new_section = True
    section = []
    reults = []
    img_num = ''; model = ''; label = ''; text_style_scale = ''; clip_loss = ''; ce_loss = ''; text_style_loss = ''
    print("Analyzing log file...")
    with open(log_file,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if line.startswith("~~~~~~~~") and end_of_new_section:
                end_of_new_section = False
                start_of_new_section = True
                section = []
            elif line.startswith("~~~~~~~~") and not end_of_new_section:
                end_of_new_section = True
                start_of_new_section = False
                img_num,model,label,text_style_scale = get_set_params(section)
            elif start_of_new_section:
                section.append(line)
            elif "clip_loss" in line:
                clip_loss = float(line.split("clip_loss_with_scale = ")[1][:-1])
            elif "ce_loss" in line:
                ce_loss = float(line.split("ce_loss = ")[1][:-1])
            elif "text_style_loss" in line:
                text_style_loss = float(line.split("text_style_loss_with_scale = ")[1][:-1])
            elif "best clip: " in line:
                captioning = line.split("best clip: ")[1]
                new_result = {"img_num":img_num, "model": model, "label":label, "text_style_scale":text_style_scale,
                              "clip_loss":clip_loss, "ce_loss":ce_loss, "text_style_loss":text_style_loss,
                              "captioning":captioning}
                reults.append(new_result)
    print(f"Finished to analyze. There are {len(reults)} params sets")
    writing_results_to_csv(reults,tgt_path_results)
    print("Finished.")



if __name__=='__main__':
    imgs_dir = 'zero-shot-style/data/imgs/'
    main()
    print("finish")