import argparse
import os.path
from argparse import ArgumentParser
import yaml
import pickle
import sys


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser()
# parser.add_argument('--epochs', type=int, default=15000, help='description')
parser.add_argument('--epochs', type=int, default=600, help='description')#todo
parser.add_argument('--lr', type=float, default=0.000001, help='description')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='description')
# parser.add_argument('--margin', type=float, default=0.4, help='description')
# parser.add_argument('--hidden_state_to_take', type=int, default=-2, help='hidden state of BERT totake')
# parser.add_argument('--last_layer_idx_to_freeze', type=int, default=-1, help='last_layer idx of BERT to freeze')
# parser.add_argument('--freeze_after_n_epochs', type=int, default=3, help='freeze BERT after_n_epochs')
parser.add_argument('--batch_size', type=int, default=28, help='description')
parser.add_argument('--inner_batch_size', type=int, default=20, help='description')
parser.add_argument('--resume', type=str, default='allow', help='continue logging to run_id')
parser.add_argument('--load_model', type=str, default='allow', help='loading best model')
parser.add_argument('--run_id', type=str, default=None, help='wandb run_id')
parser.add_argument('--tags', type=str, nargs='+', default=None, help='wandb tags')
parser.add_argument('--data_file', type=str, default='preprocessed_data.csv', help='')
parser.add_argument('--experiment_name', type=str, default='tmp', help='experiment identifier. results and checkpoints will be saved under directories with this name')
parser.add_argument('--results_dir', type=str, default=os.path.join(os.path.expanduser('~'), 'results'), help='results path')
parser.add_argument('--checkpoints_dir', type=str, default=os.path.join(os.path.expanduser('~'), 'checkpoints'), help='checkpoints path')
parser.add_argument('--data_name', type=str, default='go_emotions', help='data to use')# Twitter/go_emotions
parser.add_argument('--desired_labels', type=str, nargs='+', default='all', help='list labels for triplet training')
parser.add_argument('--override', type=str2bool, default=False, help='override results without warning')
#parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.expanduser('~'), 'projects', 'zero-shot-style', 'zero_shot_style', 'data'), help='data path')
parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.expanduser('~'), 'data'), help='data path')
#parser.add_argument('--wandb_mode', type=str, default='online', help='disabled, offline, online')
parser.add_argument('--wandb_mode', type=str, default='disabled', help='disabled, offline, online')
# parser.add_argument('--config_file', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'configs','default_config.yaml'), help='full path to config file')

#parser.add_argument('--config_file', type=str, default=os.path.join('.', 'configs','emotions_config_all_classes.yaml'), help='full path to config file')
#parser.add_argument('--config_file', type=str, default=os.spath.join('.', 'configs','twitter_config.yaml'), help='full path to config file')
#parser.add_argument('--config_file', type=str, default=os.path.join('.',  'configs','flickrstyle10k_config.yaml'), help='full path to config file')
parser.add_argument('--config_file', type=str, default=os.path.join('.',  'configs','senticap_config.yaml'), help='full path to config file')


# parser.add_argument('--config_file', type=str, default=os.path.join('..',  'configs','text_style_classification.yaml'), help='full path to config file')

parser.add_argument('--plot_only_clustering', type=str2bool, default=False, help='plot only clustering of the best model')
# parser.add_argument('--rundry', type=str2bool, default=False)
parser.add_argument('--mean_vec_emb_file', type=str, default=None, help='path to pickle file store the vec embedding')
parser.add_argument('--median_vec_emb_file', type=str, default=None, help='path to pickle file store the vec embedding')

# parser.add_argument("--beam_size", type=int, default=5)
# parser.add_argument("--num_iterations", type=int, default=5)
# parser.add_argument("--clip_scale", type=float, default=1)
# parser.add_argument("--ce_scale", type=float, default=0.2)
# parser.add_argument("--text_style_scale", type=float, default=1)


def update_hparams(hparams, args):
    # override default hparams with specified system args
    # prioritization: 0 (highest) - specified system args, 1 - yaml, 2 - parser defaults.
    # todo - first priority: sys.args second - yaml, third - parser default
    for k, v in vars(args).items():
        if k in [x[2:] if x.startswith('--') else x for x in sys.argv]:
            hparams[k] = v
        elif k in hparams.keys():  # means k exists in yaml
            # don't do anything
            pass
        else:
            # take parser's default
            hparams[k] = v
        # if k not in hparams.keys() or parser.get_default(k) != v:
        #     hparams[k] = v
    return hparams


def get_hparams(args):
    # read data specifications
    # with open(args.data_config) as f:
    #     data_config = yaml.load(f, Loader=yaml.FullLoader)

    # read default experiment config from yaml
    print(f'config_file: {args.config_file}')
    if args.config_file.endswith('.yaml'):
        with open(args.config_file) as f:
            experiment_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        assert args.configfile.endswith('.pkl')
        with open(args.configfile, 'rb') as f:
            experiment_config = pickle.load(f)
    # general hparam dict for all modules
    # hparams = {**experiment_config, **data_config}  # combine dictionaries (latter overrides the former values if same k in both)

    # update hparams with system args
    # hparams = update_hparams(hparams, args)
    hparams = update_hparams(experiment_config, args)
    return hparams