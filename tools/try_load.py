import argparse

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from trainer import Trainer
from trainer import setup

ap = argparse.ArgumentParser()
ap.add_argument('config_file')
args = ap.parse_args()


# cfg = get_cfg()
# cfg.merge_from_file(args.config_file)
cfg = setup(args, freeze=False)

# cfg.MODEL.WEIGHTS='weights/coco_personkp/model_final_a6e10b.pkl'
# cfg.MODEL.WEIGHTS = 'weights/coco_personkp/model_final_a6e10b_anchor-removed.pkl'
cfg.MODEL.WEIGHTS = 'weights/mask_deform/model_final_821d0b.pkl'
cfg.MODEL.LOAD_PROPOSALS = False
cfg.DATASETS.TRAIN = ()
cfg.freeze()

model = Trainer.build_model(cfg)
print(cfg.MODEL.WEIGHTS)
DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)

pthfile = cfg.MODEL.WEIGHTS
import pickle
with open(pthfile, "rb") as f:
    data = pickle.load(f, encoding="latin1")
