import pickle

import yaml

from zero_shot_style.utils import parser, get_hparams
import os
# source_pickle_file_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/flickrstyle10k_fixed_param_25_3_23/prompt_manipulation/19i488ig-ancient-brook-51/config.pkl'
# tgt_file_path = '/Users/danielabendavid/experiments/stylized_zero_cap_experiments/flickrstyle10k_fixed_param_25_3_23/prompt_manipulation/19i488ig-ancient-brook-51/restore_cnfig.yaml'
#
# config_baseline.yaml
# config_embedding.yaml
# config_to_check.yaml
# /Users/danielabendavid/projects/zero-shot-style/zero_shot_style/configs/check_remove/config_baseline.yaml
def get_args():
    parser.add_argument('--config_file', type=str,
                        default=os.path.join('.', 'configs', 'config.yaml'),
                        help='full path to config file')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    base_dir = '/Users/danielabendavid/projects/zero-shot-style/zero_shot_style/configs/check_remove'
    src_files = ['config_baseline.yaml','config_embedding.yaml','config_to_check.yaml']
    for f in src_files:
        file_path = os.path.join(base_dir,f)
        tgt_path = os.path.join(base_dir,'restored_'+f)
        args.config_file = file_path
        config = get_hparams(args)
        with open(tgt_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    print('finish')