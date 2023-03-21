import os
import shutil
src_dir = '/Users/danielabendavid/data/senticap/images/test'
tgt_dir = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/human_evaluation/images_for_human_evaluation'
images_for_human_evaluation = [38488, 48050, 60409, 72873, 108698, 132796, 152740, 153966, 154420, 160531, 199050, 252610, 356002, 387431, 407260, 462635, 478798, 480793, 511999, 561437]

for im in os.listdir(src_dir):
    if im[0]!='.' and int(im.split('.')[0]) in images_for_human_evaluation:
        shutil.copyfile(os.path.join(src_dir,im),os.path.join(tgt_dir,im))
print('finished to copy the relevant files.')
print('finish')