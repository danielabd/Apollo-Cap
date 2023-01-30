import os.path
import pickle

import pandas as pd


def main():
    pkl_files = [os.path.join(os.path.expanduser('~'),'data/senticap/annotations','train.pkl')]
    pkl_files.append(os.path.join(os.path.expanduser('~'),'data/senticap/annotations','val.pkl'))
    pkl_files.append(os.path.join(os.path.expanduser('~'),'data/senticap/annotations','test.pkl'))
    target_pth_caption = os.path.join(os.path.expanduser('~'),'data/senticap/annotations','all_caption.csv')
    data_type = []
    img_name_list = []
    factual_captions = []
    pos_captions = []
    neg_captions = []
    img_path = []
    for pkl_file in pkl_files:
        with open(pkl_file,'rb') as fp:
            data = pickle.load(fp)
        for d in data:
            data_type.append(pkl_file.split('/')[-1].split('.')[0])
            img_name_list.append(int(data[d]['image_path'].split('/')[-1].split('.')[0]))
            img_path.append(data[d]['image_path'])
            factual_captions.append(data[d]['factual'])
            pos_captions.append(data[d]['positive'])
            neg_captions.append(data[d]['negative'])
    total_data = {'data type': data_type, 'img_name': img_name_list, 'factual_captions': factual_captions,'pos_captions': pos_captions, 'neg_captions': neg_captions, 'img_path': img_path}
    df = pd.DataFrame(total_data)
    df.to_csv(target_pth_caption)
    print('finish')

if __name__=='__main__':
    main()