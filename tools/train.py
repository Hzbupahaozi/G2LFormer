#!/usr/bin/env python
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config,DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmocr import __version__
from mmocr.apis import train_detector
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.utils import collect_env, get_root_logger


torch.multiprocessing.set_start_method('fork')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector.')
    parser.add_argument('config', default='/home/chs/tablemaster-mmocr/configs/textrecog/master/table_master_ConcatLayer_ResnetExtract_Ranger_wtw.py', help='Train config file path.')
    parser.add_argument('--work-dir', default='/home/chs/tablemaster-mmocr/work_dir_chs', help='The dir to save logs and models.')
    parser.add_argument(
        '--load-from', help='The checkpoint file to load from.')
    parser.add_argument(
        '--resume-from', help='The checkpoint file to resume from.')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Whether not to evaluate the checkpoint during training.')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of gpus to use '
        '(only applicable to non-distributed training).')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be of the form of either '
        'key="[a,b]" or key=a,b .The argument also allows nested list/tuple '
        'values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks '
        'are necessary and that no white space is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Options for job launcher.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--mc-config',
        type=str,
        default='',
        help='Memory cache config for image loading speed-up during training.')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options
    # args.gpu_ids = [1]
    # args.no_validate = False
    return args


def main():
    
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # update mc config
    if args.mc_config:
        mc = Config.fromfile(args.mc_config)
        if isinstance(cfg.data.train, list):
            for i in range(len(cfg.data.train)):
                cfg.data.train[i].pipeline[0].update(
                    file_client_args=mc['mc_file_client_args'])
        else:
            cfg.data.train.pipeline[0].update(
                file_client_args=mc['mc_file_client_args'])

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)
    # print(cfg.model)
    # print(cfg.get('train_cfg'))
    # print(cfg.get('test_cfg'))
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # print(model)
    # 查看模型参数
    # total = sum(p.numel() for p in model.parameters())
    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('total:', total)
    # print('trainable:', trainable)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmocr_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
