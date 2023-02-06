import csv
import os

import pandas as pd

def main():
    src_dir = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/5_2_23/final_test/text_style'
    src_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/5_2_23/final_test/text_style/4t8F8naAqLgd8af5nL2KHy-dummy-4t8F8naAqLgd8af5nL2KHy/results_all_models_source_classes_23_26_35__05_02_2023.csv'
    tgt_path ='/Users/danielabendavid/experiments/stylized_zero_cap_experiments/5_2_23/final_test/text_style/4t8F8naAqLgd8af5nL2KHy-dummy-4t8F8naAqLgd8af5nL2KHy/fixed_results_all_models_source_classes_23_26_35__05_02_2023.csv'
    src_dir = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/5_2_23/final_test/image_manipulation'
    for d in os.listdir(src_dir):
        if os.path.isdir(os.path.join(src_dir,d)):
            for f in os.listdir(os.path.join(src_dir,d)):
                if f.endswith('.csv'):
                    src_path = os.path.join(src_dir,d,f)
                    tgt_path = os.path.join(src_dir,d,'results.csv')

                    pd_data = pd.read_csv(src_path)
                    with open(src_path,'r') as fp:
                        lines = fp.readlines()
                    data = {}
                    data_csv = {'img_num':[],'positive':[],'negative':[]}
                    img_num_list = []
                    pos_list = []
                    neg_list = []
                    for i,line in enumerate(lines):
                        if i > pd_data.shape[0]:
                            break
                        if line.startswith('img_num'):
                            img_num = line[len('img_num'):].split(',')[1]
                            data[img_num] = {'img_num':img_num}
                            img_num_list.append(img_num)
                            data_csv['img_num'].append(img_num)
                            continue
                        if line.startswith('senticap'):
                            data[img_num]['positive'] = pd_data.iloc[i-1,1]
                            data[img_num]['negative'] = pd_data.iloc[i-1,2]
                            pos_list.append(pd_data.iloc[i-1,1])
                            neg_list.append(pd_data.iloc[i-1,2])
                            data_csv['positive'].append(pd_data.iloc[i-1,1])
                            data_csv['negative'].append(pd_data.iloc[i-1,2])
                    rows=[data[img_num] for img_num in data]
                    fieldnames = ['img_num', 'positive', 'negative']
                    with open(tgt_path, 'w', encoding='UTF8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)



    print('finish')

if __name__=='__main__':
    main()