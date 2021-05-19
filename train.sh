#!/bin/bash
# python3 train_net.py --num-gpus $NUM_GPU --config-file $CONFIG \
# OUTPUT_DIR $OUT_DIR \
# MODEL.WEIGHTS $MODEL_WEIGHTS \
# DATASETS.TRAIN $DATASETS_TRAIN \
# DATASETS.TEST $DATASETS_TEST 


## Quick run to test all working
# python3 train_net_clearml.py --num-gpus 1 --awskey Quicksilver123@  --config-file configs/persdet/persdet_rcnn_R_50_FPN_ECP_day.yaml --model-weights faster_rcnn_R_50_FPN_3x/model_final_280758.pkl --output-dir ./output --datasets-train "('MOT20_indoor_train',)" --datasets-test "('MOT20_indoor_val',)" --download-data MOT20_indoor.tar.gz  \
# --test-eval-period 100 \
# --solver-ims-per-batch 1 \
# --solver-base-lr 0.01 \
# --solver-warmup-iters 10 \
# --solver-steps "(100,)" \
# --solver-max-iter 200 \
# --solver-checkpoint-period 100


## ECP Day
# python3 train_net_clearml.py --num-gpus 1 --awskey Quicksilver123@  --config-file configs/persdet/persdet_rcnn_R_50_FPN_ECP_day.yaml --model-weights faster_rcnn_R_50_FPN_3x/model_final_280758.pkl --output-dir ./output --datasets-train "('ECP_day_train',)" --datasets-test "('ECP_day_val',)" --download-data ECP_day.tar.gz  \
# --test-eval-period 5000 \
# --solver-ims-per-batch 4 \
# --solver-base-lr 0.01 \
# --solver-warmup-iters 1000 \
# --solver-steps "(50000,100000,150000)" \
# --solver-max-iter 200000 \
# --solver-checkpoint-period 5000

python3 train_net_clearml.py --clearml-task-name WIDER_st_pt_ECP_biggie --clearml-task-type training  --num-gpus 1 --awskey Quicksilver123@  --config-file configs/persdet/persdet_rcnn_R_50_FPN_biggie.yaml --model-weights ECP/model_0119999.pth --output-dir ./output --datasets-train "('WiderPedestrian_street_train',)" --datasets-test "('WiderPedestrian_street_val',)" --download-data WiderPedestrian_street.tar.gz --test-eval-period 5000 --solver-ims-per-batch 2 --solver-base-lr 0.01 --solver-warmup-iters 1000 --solver-steps "(50000,100000,150000)" --solver-max-iter 200000 --solver-checkpoint-period 5000
