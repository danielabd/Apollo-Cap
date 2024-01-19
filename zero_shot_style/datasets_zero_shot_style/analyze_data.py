import json
import os.path
import pickle

import pandas as pd

def get_factual_captions(factual_file_path_list):
    '''
    get dict of all factual captions
    :param factual_file_path_list: list of file path to factual data
    :return: factual_captions:dict:key=img_name,value=list of factual captions
    '''
    print("create dict of factual captions...")
    factual_captions = {}
    finish_im_ids = []
    for factual_file_path in factual_file_path_list:
        data = json.load(open(factual_file_path, "r"))
        for d in data['annotations']:
            if d['image_id'] in finish_im_ids:
                continue
            if d['image_id'] not in factual_captions:
                factual_captions[d['image_id']] = [d['caption']]
            else:
                factual_captions[d['image_id']].append(d['caption'])
        finish_im_ids = list(factual_captions.keys())
    return factual_captions


def main():
    data_dir = os.path.join(os.path.expanduser('~'),'data')
    pkl_files = [os.path.join(data_dir,'senticap/annotations','train.pkl')]
    pkl_files.append(os.path.join(data_dir,'senticap/annotations','val.pkl'))
    pkl_files.append(os.path.join(data_dir,'senticap/annotations','test.pkl'))
    target_pth_caption = os.path.join(data_dir,'senticap/annotations','all_caption.csv')
    factual_file_path_list = [
        os.path.join(data_dir, 'source', 'coco', '2014', 'annotations', 'captions_train2014.json'),
        os.path.join(data_dir, 'source', 'coco', '2014', 'annotations', 'captions_val2014.json'),
        os.path.join(data_dir, 'source', 'coco', '2017', 'annotations', 'captions_train2017.json'),
        os.path.join(data_dir, 'source', 'coco', '2017', 'annotations', 'captions_val2017.json')]
    factual_captions_path = os.path.join(data_dir,'source','coco','factual_captions.pkl')

    if os.path.exists(factual_captions_path):
        with open(factual_captions_path, 'rb') as f:
            factual_captions_dict = pickle.load(f)
    else:
        factual_captions_dict = get_factual_captions(factual_file_path_list)
        with open(factual_captions_path, 'wb') as f:
            pickle.dump(factual_captions_dict, f)

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
            img_name = int(data[d]['image_path'].split('/')[-1].split('.')[0])
            img_name_list.append(int(data[d]['image_path'].split('/')[-1].split('.')[0]))
            img_path.append(data[d]['image_path'])
            # factual_captions.append(data[d]['factual'])
            factual_captions.append(factual_captions_dict[img_name])
            pos_captions.append(data[d]['positive'])
            neg_captions.append(data[d]['negative'])
    total_data = {'data type': data_type, 'img_name': img_name_list, 'factual_captions': factual_captions,'pos_captions': pos_captions, 'neg_captions': neg_captions, 'img_path': img_path}
    df = pd.DataFrame(total_data)
    df.to_csv(target_pth_caption)
    print('finish')

if __name__=='__main__':
    main()