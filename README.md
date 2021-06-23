# ClearML training works!

[`detectron2`](https://github.com/facebookresearch/detectron2) training with [`ClearML`](https://github.com/allegroai/clearml) hooks.

## Usage

Training script: `./train_net_clearml.py` 

Base parameters are defined in the given config yaml file, but argparse arguments will override them. The inclusion of hyperparams into `argparse` is for smooth integration with `ClearML`'s hyperparam (magic) tracking.
