# ClearML training works!

[`detectron2`](https://github.com/facebookresearch/detectron2) training with [`ClearML`](https://github.com/allegroai/clearml) hooks.

## Usage

Training script: `./train_net_clearml.py` but this comes with many argparse arguments

Base parameters are defined in the given config yaml file, but argparse arguments will override them. 

Please see `./train.sh` for a guide on how to write these args. 
