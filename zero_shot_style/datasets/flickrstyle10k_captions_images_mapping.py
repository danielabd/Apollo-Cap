#https://uofi.app.box.com/s/p39twkfbves4mhsk7msh5eypznqqqbq5/file/1046610063096
import os
from sklearn.model_selection import train_test_split


class ImageFlickrStyle10k:
    def __init__(self,filename):
        self.filename = filename
        self.imgpath = ''
        self.imgid = None
        self.humor = []
        self.romantic = []
        self.factual = []
        self.split = None#    TEST_SPLIT = 0,   TRAIN_SPLIT = 1,    VAL_SPLIT = 2

    def set_filename(self,filename):
        self.filename = filename

    def set_imgpath(self,path):
        self.imgpath = path

    def set_imgid(self,id):
        self.imgid = id

    def add_sentence(self,style,sentence):
        if style == 'humor':
            if type(sentence) == list:
                self.humor.extend(sentence)
            else:
                self.humor.append(sentence)
        if style == 'romantic':
            if type(sentence) == list:
                self.romantic.extend(sentence)
            else:
                self.romantic.append(sentence)
        if style == 'factual':
            if type(sentence) == list:
                self.factual.extend(sentence)
            else:
                self.factual.append(sentence)

    def set_imgpath_by_dir_path(self, imgs_folder, suffix):
        self.imgpath = os.path.join(imgs_folder,self.filename+suffix)

    def get_sentences(self, style):
        if style=='humor':
            return self.humor
        if style=='romantic':
            return self.romantic
        if style=='factual':
            return self.factual

    def get_filename(self):
        return self.filename

    def get_imgpath(self):
        return self.imgpath

    def get_imgid(self):
        return self.imgid


def mapping_data(base_path,styles):
    '''

    :param base_path: base path to start from
    :param styles: list of required styles
    :return:
    '''
    stylized_mapping_idx_to_img_name = {}
    for style in styles:
        stylized_mapping_idx_to_img_name[style] = {}
        with open(os.path.join(base_path,f'FlickrStyle_v0.9/{style}/train.p')) as fp:
            data = fp.readlines()
        idx = -1
        for line in data:
            if '.' not in line: #idx of img
                idx = int(line.split('p')[1].split('\n')[0])
            else: #img name
                img_name = line.split('.')[0]
                if idx in stylized_mapping_idx_to_img_name[style]:
                    print("idx in stylized_mapping_idx_to_img_name[style] - check it!")
                if img_name in stylized_mapping_idx_to_img_name[style].values():
                    print("img_name in stylized_mapping_idx_to_img_name[style].values() - check it!")
                stylized_mapping_idx_to_img_name[style][idx] = img_name[2:]
        print(f'Finihsed to map {style}')
    return stylized_mapping_idx_to_img_name


def get_stylized_captions(lines, mapping_idx_to_img_name):
    stylized_captions = {}  # key=image name, value=caption
    for i,line in enumerate(lines):
            stylized_captions[mapping_idx_to_img_name[i+1]] = line.split('\n')[0]
    return stylized_captions


def get_factual_captions(lines):
    factual_captions = {}
    for line in lines[1:]:
        data = line.split(',')
        img_name = data[0].split('.')[0]
        if img_name not in factual_captions:
            factual_captions[img_name] = [data[1].split('\n')[0]]
        else:
            factual_captions[img_name].append(data[1].split('\n')[0])
    return factual_captions


def get_captions(stylized_mapping_idx_to_img_name, captions_file_path, styles='all'):
    '''

    :param stylized_mapping_idx_to_img_name: dict. key=style. value=dict: key=idx. value=image name.
    :return: captions - dict with key=image name, value = list of captions
    '''
    captions = {}
    if styles=='all':
        styles_to_check = list(captions_file_path.keys())
    for style in styles_to_check:
        with open(captions_file_path[style], encoding= 'unicode_escape') as fp:
            lines = fp.readlines()
        if style=='factual':
            captions[style] = get_factual_captions(lines[1:])
        else: #stylized
            captions[style] = get_stylized_captions(lines, stylized_mapping_idx_to_img_name[style])
    merged_captions = merge_all_captions(captions)  # dict with image name in the key
    return merged_captions


def merge_all_captions(captions):
    '''
    merge all captions by image name in the key
    :param captions: dict: key=style, value=dict:key=image_name,value=caption
    :return: merged_captions: dict:key=image_name, value=ImageFlickrStyle10k
    '''
    merged_captions = {}
    styles = list(captions.keys())
    intersection_set = set(captions[styles[0]].keys())
    for s in styles[1:]:
        set1 = set(captions[s].keys())
        intersection_set = set1.intersection(intersection_set)
    #intersection_set have only the keys shared in all styles (include factual)
    for k in intersection_set:
        merged_captions[k] = ImageFlickrStyle10k(k)
        for s in captions.keys():
            merged_captions[k].add_sentence(s,captions[s][k])
    return merged_captions


def set_image_path(merged_captions,imgs_folder):
    for k in merged_captions:
        merged_captions[k].set_imgpath_by_dir_path(imgs_folder,'.jpg')


def get_all_flickrstyle10k_data(base_path,imgs_folder,captions_file_path):
    stylized_mapping_idx_to_img_name = mapping_data(base_path, ['humor', 'romantic'])
    merged_captions = get_captions(stylized_mapping_idx_to_img_name, captions_file_path, 'all')
    set_image_path(merged_captions, imgs_folder)
    print("Finished to create all flickrstyle10k data")
    return merged_captions

#def get_flickrstyle_images_path_to_train():
def main():
    base_path = '../../../../data/source/flickrstyle10k'
    imgs_folder = '../../../../data/source/flickrstyle10k/flickr_images_dataset'
    captions_file_path= {'humor':os.path.join(base_path,'FlickrStyle_v0.9/humor/funny_train.txt'),
                         'romantic':os.path.join(base_path,'FlickrStyle_v0.9/romantic/romantic_train.txt'),
                         'factual':os.path.join(base_path,'FlickrStyle_v0.9/captions.txt')}
    stylized_mapping_idx_to_img_name = mapping_data(base_path,['humor','romantic'])
    merged_captions = get_captions(stylized_mapping_idx_to_img_name,captions_file_path,'all')
    set_image_path(merged_captions,imgs_folder)
    #merged_captions_keys = list(merged_captions.keys())
    #train_data, val_test_data = train_test_split(merged_captions_keys, test_size=0.15, random_state=42)
    #val_test, test_test = train_test_split(val_test_data, test_size=0.15, random_state=42)
    # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),  # todo check sklearn split data func - keeps proportions between classes across all splits
    #                                      [int(.8 * len(df)), int(.9 * len(df))])
    print('Finished mains')


if __name__=='__main__':
    main()