"""
Rank commandline arguments.
"""
import configargparse

from . import post_args


def get_general_args(cmd=None):
    """
    Create an ArgParser object and add general arguments to it.

    Return ArgParser object.
    """
    if cmd is None:
        cmd = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    cmd.add('--tree_type', type=str, nargs='+', default=['lgb', 'sgb', 'xgb', 'cb'])
    return cmd


def get_ranking_args():
    """
    Add arguments specific to "Ranking" plot.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/')

    # ROAR and CF
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])

    # ROAR, poison, and noise
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5])

    # CF only
    cmd.add('--step_size', type=int, default=10)

    # noise and poison
    cmd.add('--val_frac', type=float, default=0.1)

    # noise only
    cmd.add('--noise_frac', type=float, default=0.4)
    cmd.add('--check_frac', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3])

    # poison only
    cmd.add('--poison_frac', type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3])

    return cmd
