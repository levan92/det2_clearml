# # training with clearml & s3
# python3 train_net_clearml.py \
# --config-file configs/det2_clearml/example.yaml \
# --clearml-proj example-proj \
# --clearml-task-name example_task_1 \
# --clearml-task-type training \
# --docker-img \
# --queue \
# --download-models \
# --s3-models-bucket \
# --s3-models-path \
# --download-data \
# --s3-data-bucket \
# --s3-data-path \
# --datasets-train \
# --datasets-test \
# --s3-output-bucket \
# --s3-output-path

# training locally completely
python3 train_net_clearml.py \
--config-file configs/det2_clearml/example.yaml \
--skip-clearml \
--skip-s3 \
--custom-dsnames coco_train coco_val \
--custom-cocojsons /media/dh/HDD/coco/annotations/instances_train2017.json /media/dh/HDD/coco/smallval/instances_val2017.json \
--custom-imgroots /media/dh/HDD/coco/train2017 /media/dh/HDD/coco/smallval/images \
--datasets-train coco_train \
--datasets-test coco_val 

# # training with clearml & s3 with multi-gpus
# python3 train_net_clearml.py \
# --config-file configs/det2_clearml/example.yaml \
# --num-gpus 4 \
# --num-machines 1 \
# --machine-rank 0 \
# --clearml-proj example-proj \
# --clearml-task-name example_task_1 \
# --clearml-task-type training \
# --docker-img \
# --queue \
# --download-models \
# --s3-models-bucket \
# --s3-models-path \
# --download-data \
# --s3-data-bucket \
# --s3-data-path \
# --datasets-train \
# --datasets-test \
# --s3-output-bucket \
# --s3-output-path