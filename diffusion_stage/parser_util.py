import argparse
import json
import os
from argparse import ArgumentParser
from yacs.config import CfgNode as CN


def get_args():
    parser = ArgumentParser(description='Train Motion Capture Network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="",
                        type=str)
    args = parser.parse_args()
    print(f"using config {args.cfg}")
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.cfg)
    cfg.name = args.cfg.split('/')[1].split('.')[0]
    if not hasattr(cfg, 'SAVE_DIR') or cfg.SAVE_DIR is None:
        cfg.SAVE_DIR = os.path.join("outputs", cfg.name)
    
    return cfg


def merge_file(args):
    print(f"using config {args.cfg}")
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.cfg)
    return cfg
