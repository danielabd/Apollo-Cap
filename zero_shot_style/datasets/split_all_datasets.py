import shutil
import pickle
from flickrstyle10k_captions_images_mapping import *
from senticap_reader import *


class ImageSentiCap:

    def __init__(self):
        self.filename = ''
        self.imgpath = ''
        self.imgid = None
        self.positive = []
        self.negative = []
        self.factual = []
        self.split = None#    TEST_SPLIT = 0,   TRAIN_SPLIT = 1,    VAL_SPLIT = 2


    def set_filename(self,filename):
        self.filename = filename

    def set_imgpath(self,path):
        self.imgpath = path

    def set_imgid(self,id):
        self.imgid = id


    def add_sentence(self,sentence,style):
        if style == 'positive':
            if type(sentence) == list:
                self.positive.extend(sentence)
            else:
                self.positive.append(sentence)
        if style == 'negative':
            if type(sentence) == list:
                self.negative.extend(sentence)
            else:
                self.negative.append(sentence)
        if style == 'factual':
            if type(sentence) == list:
                self.factual.extend(sentence)
            else:
                self.factual.append(sentence)

    def set_imgpath_by_dir_path(self, imgs_folder, suffix):
        self.imgpath = os.path.join(imgs_folder,self.filename,suffix)

    def get_sentences(self, style):
        if style=='positive':
            return self.positive
        if style=='negative':
            return self.negative
        if style=='factual':
            return self.factual

    def get_filename(self):
        return self.filename

    def get_imgpath(self):
        return self.imgpath

    def get_imgid(self):
        return self.imgid

def add_factual_sentences_to_senticap_data(sr,factual_file_path_list):
    '''

    :param sr:
    :param factual_file_path_list: list of file path to factual data
    :return:
    '''
    print("Adding factual sentences to senticap data...")

    factual_captions = {}
    for factual_file_path in factual_file_path_list:
        data = json.load(open(factual_file_path, "r"))
        for d in data['annotations']:
            if d['image_id'] not in factual_captions:
                factual_captions[d['image_id']] = [d['caption']]
            else:
                factual_captions[d['image_id']].append(d['caption'])

    senticap_captions_test = []
    num_skip_imgs_because_styles = 0
    senticap_captions_train = []
    for s in sr.get_images():
        senticap_image = ImageSentiCap()
        img_id = s.getImgID()
        senticap_image.set_imgid(img_id)
        senticap_image.set_filename(s.getFilename)
        senticap_image.set_imgpath(s.get_imgpath())
        sentiments = []
        for sen in s.getSentences():
            if sen.getSentimentPolarity():
                sentiment_polarity = 'positive'
            else:
                sentiment_polarity = 'negative'
            sentiments.append(sentiment_polarity)
            raw_sentence = sen.getRawsentence()
            senticap_image.add_sentence(sen.getRawsentence(),sentiment_polarity)
        # we take for test set only images which have factual and at least one of positive or negative sentences. suppose for training we don't need a factual caption
        if 'positive' or 'negative' in sentiments:
            if img_id in factual_captions:
                senticap_image.add_sentence(factual_captions[img_id], 'factual')
                senticap_captions_test.append(senticap_image)
                continue
        if 'positive' and 'negative' not in sentiments:
            num_skip_imgs_because_styles += 1
            continue
        #take data for training
        senticap_captions_train.append(senticap_image)
    print(f'{num_skip_imgs_because_styles} does not have both positive and negative captions')
    return senticap_captions_test, senticap_captions_train


def save_senticap_data(dir_path, train_data, val_data, test_data):
    data_dict = {}
    for da in train_data:
        filename = da.imgpath.split('/')[-1].split('.')[0]
        data_dict[filename] = {"factual": da.factual, 'positive': da.positive, 'negative': da.negative, 'image_path': da.imgpath}
    with open(os.path.join(dir_path, 'train.pkl'), 'wb') as file:
        pickle.dump(data_dict, file)
    data_dict = {}
    for da in val_data:
        filename = da.imgpath.split('/')[-1].split('.')[0]
        data_dict[filename] = {"factual": da.factual, 'positive': da.positive, 'negative': da.negative, 'image_path': da.imgpath}
    with open(os.path.join(dir_path, 'val.pkl'), 'wb') as file:
        pickle.dump(data_dict, file)
    data_dict = {}
    for da in test_data:
        filename = da.imgpath.split('/')[-1].split('.')[0]
        data_dict[filename] = {"factual": da.factual, 'positive': da.positive, 'negative': da.negative, 'image_path': da.imgpath}
    with open(os.path.join(dir_path, 'test.pkl'), 'wb') as file:
        pickle.dump(data_dict, file)

def save_flickrstyle10k_data(dir_path, train_data, val_data, test_data):
    data_dict = {}
    for da in train_data:
        data_dict[da.filename] = {"factual": da.factual, 'humor': da.humor, 'romantic': da.romantic, 'image_path': da.imgpath}
    with open(os.path.join(dir_path, 'train.pkl'), 'wb') as file:
        pickle.dump(data_dict, file)
    data_dict = {}
    for da in val_data:
        data_dict[da.filename] = {"factual": da.factual, 'humor': da.humor, 'romantic': da.romantic, 'image_path': da.imgpath}
    with open(os.path.join(dir_path, 'val.pkl'), 'wb') as file:
        pickle.dump(data_dict, file)
    data_dict = {}
    for da in test_data:
        data_dict[da.filename] = {"factual": da.factual, 'humor': da.humor, 'romantic': da.romantic, 'image_path': da.imgpath}
    with open(os.path.join(dir_path, 'test.pkl'), 'wb') as file:
        pickle.dump(data_dict, file)


def arrange_data(train_data,val_data, test_data,target_dir, save_images, save_annotations, dataset_name):
    print("Arranging the data...")
    #copy images
    if save_images:
        for d in train_data:
            shutil.copyfile(d.imgpath,os.path.join(target_dir,'images','train',d.imgpath.split('/')[-1]))
        for d in val_data:
            shutil.copyfile(d.imgpath,os.path.join(target_dir,'images','val',d.imgpath.split('/')[-1]))
        for d in test_data:
            shutil.copyfile(d.imgpath,os.path.join(target_dir,'images','test',d.imgpath.split('/')[-1]))
    #save annotations
    if save_annotations:
        annotation_dir = os.path.join(target_dir,'annotations')
        if dataset_name == 'flickrstyle10k':
            save_flickrstyle10k_data(annotation_dir, train_data, val_data, test_data)
        if dataset_name == 'senticap':
            save_senticap_data(annotation_dir, train_data, val_data, test_data)



def main():
    #flickrstyle10k
    base_path_flickrstyle10k = '../../../../data/source/flickrstyle10k'
    imgs_folder_flickrstyle10k = '../../../../data/source/flickrstyle10k/flickr_images_dataset'
    captions_file_path_flickrstyle10k= {'humor':os.path.join(base_path_flickrstyle10k,'FlickrStyle_v0.9/humor/funny_train.txt'),
                         'romantic':os.path.join(base_path_flickrstyle10k,'FlickrStyle_v0.9/romantic/romantic_train.txt'),
                         'factual':os.path.join(base_path_flickrstyle10k,'FlickrStyle_v0.9/captions.txt')}
    #senticap
    filename_senticap = '../../../../data/source/senticap/senticap_dataset/data/senticap_dataset.json'
    imgs_folder_senticap = '../../../../data/source/coco/2014'
    imgs_folder2017 = '../../../../data/source/coco/2017/images'

    target_data_dir_flickrstyle10k = os.path.abspath('../../../../data/flickrstyle10k')
    factual_file_path_list_senticap = [
        '/Users/danielabendavid/data/source/coco/2014/annotations/captions_train2014.json',
        '/Users/danielabendavid/data/source/coco/2014/annotations/captions_val2014.json']
    target_data_dir_senticap = os.path.abspath('../../../../data/senticap')
    flickrstyle10k_data = get_all_flickrstyle10k_data(base_path_flickrstyle10k,imgs_folder_flickrstyle10k,captions_file_path_flickrstyle10k)
    flickrstyle10k_data_list = list(flickrstyle10k_data.values())


    sr = SenticapReader(filename_senticap,imgs_folder_senticap,imgs_folder2017)
    senticap_captions_test, senticap_captions_train = add_factual_sentences_to_senticap_data(sr,factual_file_path_list_senticap)


    ## split datat to train,val,test
    # flickrstyle10k
    train_data_flickrstyle10k, val_test_data_flickrstyle10k = train_test_split(flickrstyle10k_data_list, test_size=0.1426, random_state=42)
    val_data_flickrstyle10k, test_data_flickrstyle10k = train_test_split(val_test_data_flickrstyle10k, test_size=0.5, random_state=42)
    #arrange_data(train_data_flickrstyle10k,val_data_flickrstyle10k, test_data_flickrstyle10k,target_data_dir_flickrstyle10k)
    # senticap
    #train_data_senticap, val_test_data_senticap = train_test_split(senticap_captions, test_size=0.15, random_state=42)
    # val_data_senticap, test_data_senticap = train_test_split( val_test_data_senticap, test_size=0.5, random_state=42)
    train_data_senticap, val_data_senticap = train_test_split(senticap_captions_train, test_size=0.1, random_state=42)
    test_data_senticap = senticap_captions_test
    # arrange_data(train_data_senticap, val_data_senticap, test_data_senticap, target_data_dir_senticap)
    save_images = False
    save_annotations = False
    #use all annotated data with factual to test
    arrange_data(train_data_flickrstyle10k, val_data_flickrstyle10k, test_data_flickrstyle10k,target_data_dir_flickrstyle10k,save_images,save_annotations, 'flickrstyle10k')
    arrange_data(train_data_senticap, val_data_senticap, test_data_senticap, target_data_dir_senticap,save_images,save_annotations, 'senticap')

    print('finish')

if __name__=='__main__':
    main()