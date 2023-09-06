import os
import pickle
import pandas as pd
import yaml


def get_list_of_imgs_for_caption(config):
    print("take list of images for captioning...")
    imgs_to_test = []
    print(f"config['max_num_of_imgs']: {config['max_num_of_imgs']}")
    # if 'specific_img_idxs_to_test' in config and len(config['specific_img_idxs_to_test']) > 0:
    #     imgs_list = os.listdir(
    #         os.path.join(os.path.join(os.path.expanduser('~'), 'data', config['dataset']), 'images',
    #                      config['data_type']))
    #     for i in config['specific_img_idxs_to_test']:
    #         i = int(i)
    #         im = imgs_list[i]
    #         imgs_to_test.append(
    #             os.path.join(os.path.join(os.path.expanduser('~'), 'data', config['dataset']), 'images',
    #                          config['data_type'], im))
    #     return imgs_to_test
    for i, im in enumerate(os.listdir(
            os.path.join(os.path.join(os.path.expanduser('~'), 'data', config['dataset']), 'images',
                         config['data_type']))):
        # if len(imgs_to_test) >= int(config['max_num_of_imgs']) > 0:
        #     break
        # if 'specific_idxs_to_skip' in config and len(config['specific_idxs_to_skip']) > 0 and i in config['specific_idxs_to_skip']:
        #     continue
        if ('.jpg' or '.jpeg' or '.png') not in im:
            continue
        # if 'specific_imgs_to_test' in config and len(config['specific_imgs_to_test']) > 0 and int(
        #         im.split('.')[0]) not in config['specific_imgs_to_test']:
        #     continue
        imgs_to_test.append(os.path.join(os.path.join(os.path.expanduser('~'), 'data', config['dataset']), 'images',
                                         config['data_type'], im))
    print(f"*** There are {len(imgs_to_test)} images to test ***")
    return imgs_to_test


def get_mapping_idx_img_name(configfile):
    with open(configfile, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # config['dataset'] = "senticap"
    imgs_to_test = get_list_of_imgs_for_caption(config)
    mapping_idx2img_name = {}
    mapping_img_name2idx = {}
    for img_path_idx, img_path in enumerate(imgs_to_test):  # img_path_list:
        if config['dataset'] == "senticap":
            img_name = int(img_path.split('.')[0].split('/')[-1])
        else: #config['dataset'] == "flickrstyle10k":
            img_name = img_path.split('.')[0].split('/')[-1]
        mapping_idx2img_name[img_path_idx] = img_name
        mapping_img_name2idx[img_name] = img_path_idx
    return mapping_idx2img_name, mapping_img_name2idx


def main():
    configfile = os.path.join('.', 'configs', 'config.yaml')

    mapping_idx2img_name, mapping_img_name2idx = get_mapping_idx_img_name(configfile)

    print("save mapping to a file")
    with open('mapping_idx_to_img_name.csv', 'w') as f:
        for key in mapping_idx2img_name.keys():
            f.write("%s,%s\n" % (key, mapping_idx2img_name[key]))
    print("finish")


if __name__=='__main__':
    main()
