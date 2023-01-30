import os.path
import pickle

import pandas as pd


def main():
    path2check = os.path.join(os.path.expanduser('~'),'experiments/stylized_zero_cap_experiments/0zpbr7nq-astral-sweep-22','config.pkl')
    with open(path2check, 'rb') as f:
        data = pickle.load(f)

    base_dir  = os.path.join(os.path.expanduser('~'),'experiments/stylized_zero_cap_experiments','30_1_2023')
    path_final_results = os.path.join(base_dir,'debug_final_results.csv')
    is_first_df = True
    first_df = pd.DataFrame()
    for d in os.listdir(base_dir):
        num = d.split('-')[-1]
        if num.isdigit() and int(num)<100:
            for f in os.listdir(os.path.join(base_dir,d)):
                if f.startswith('avg'):
                    # with open(os.path.join(base_dir,d,f)) as fp:
                    #     fp.readlines()
                    new_df = pd.read_csv(os.path.join(base_dir,d,f))
                    new_df.insert(0,"sweep_num",[int(num)]*new_df.shape[0])
                    # if new_df['img_path'][0].split('/')[-1].split('.jpg')[0]=='000000415413' and int(num)==15:
                    #     print('check')
                    # if new_df['caption']=='Image of a passenger on the train in which he was killed.' and new_df['img_path'][0].split('/')[-1].split('.jpg')[0]=='000000429063' and int(num)==22:
                    if new_df['caption'][0]=='Image of a passenger on the train in which he was killed.' and new_df['img_path'][0].split('/')[-1].split('.jpg')[0]=='000000429063':
                        #sweep = '0zpbr7nq-astral-sweep-22'
                        print('check')
                    if is_first_df:
                        first_df = new_df
                        is_first_df = False
                    else:
                        first_df = first_df.append(new_df, ignore_index=True)
                    # print(first_df.to_string())
    print(f'write final results to: {path_final_results}')
    first_df.to_csv(path_final_results)
    print('finish')


if __name__=='__main__':
    main()