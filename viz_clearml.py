# this needs to be here for it to read your args
from clearml import Task, Logger

import argparse
from pathlib import Path

CLEARML_PROJECT_NAME = 'persdet2'
IMG_EXTS = ['.jpg','.jpeg','.png','.JPG','.JPEG','.PNG']

parser = argparse.ArgumentParser()
parser.add_argument("--clearml-task-name", default="Viz Task", help="ClearML Task Name")
parser.add_argument("--clearml-task-type", default="inference", help="ClearML Task Type, e.g. training, testing, inference, etc", choices=['training','testing','inference','data_processing','application','monitor','controller','optimizer','service','qc','custom'])
parser.add_argument(
    "--awskey",
    help="Key to S3 bucket"
)

parser.add_argument(
    "--download-data",
    help="Dataset to download",
)
parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")

parser.add_argument(
    "--model-weights",
    help="MODEL.WEIGHTS"
)
parser.add_argument(
    "--bs",
    help="Inference batch size",
    default=4,
    type=int
    )
parser.add_argument(
    "--numclasses",
    help="Num of inference classes",
    default=1,
    type=int
    )
parser.add_argument(
    "--thresh",
    help="Inference confidence threshold",
    default=0.5,
    type=float
    )
parser.add_argument(
    "--min-size-test",
    help="MIN_SIZE_TEST",
    type=int
    )
parser.add_argument(
    "--max-size-test",
    help="MAX_SIZE_TEST",
    type=int
    )
args = parser.parse_args()

task = Task.init(project_name=CLEARML_PROJECT_NAME,task_name=args.clearml_task_name, task_type=args.clearml_task_type)
task.set_base_docker("harbor.io/custom/detectron2:v3 --env GIT_SSL_NO_VERIFY=true --env TRAINS_AGENT_GIT_USER=testuser --env TRAINS_AGENT_GIT_PASS=testuser" )
task.execute_remotely(queue_name="gpu", exit_process=True)
logger = task.get_logger()


'''
S3 downloading
'''
import boto3
from botocore.client import Config
import tarfile

def download_dir_s3(s3_resource, bucket_name, remote_dir_subpath, local_dir_path):
    bucket = s3_resource.Bucket(bucket_name)
    downloaded_dirs = []
    for obj in bucket.objects.filter(Prefix=str(remote_dir_subpath)):
        local = local_dir_path / obj.key
        local.parent.mkdir(exist_ok=True, parents=True)
        if local.parent not in downloaded_dirs:
            downloaded_dirs.append(local.parent)
        bucket.download_file(obj.key, str(local))
    return downloaded_dirs

s3=boto3.resource('s3', 
        endpoint_url='http://192.168.56.253:9000/',
        aws_access_key_id='lingevan',
        aws_secret_access_key=args.awskey,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1')

datasets_bucket = 'datasets'
local_data_dir = Path('datasets')
if args.download_data:
    download_data = Path(args.download_data)
    print(f'Downloading {download_data} from S3..')
    downloaded_dirs = download_dir_s3(s3, datasets_bucket, download_data, local_data_dir)
    print(f'Datasets: {args.download_data} downloaded from S3!')

if args.model_weights:
    magic_weights_path = Path('cv-models/persdet/det2')
    s3_weights_path = magic_weights_path / Path(args.model_weights)
    local_weights_path = 'weights' / Path(args.model_weights)
    local_weights_path.parent.mkdir(parents=True, exist_ok=True)
    s3.Bucket('models').download_file(str(s3_weights_path), str(local_weights_path))

    assert local_weights_path.is_file()
    print(f'Weights: {args.model_weights} downloaded from S3!')


'''
INFERENCE
'''
import cv2
import torch 
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

def batch(iterable, bs=1):
    length = len(iterable)
    for ndx in range(0,length,bs):
        yield iterable[ndx:min(ndx+bs,length)]

def preproc(img, transform_gen):
    if to_flip_channel:
        img = img[:,:,::-1]
    image = transform_gen.get_transform(img).apply_image(img)
    image = torch.as_tensor(image.astype("float32").transpose(2,0,1))
    return image

cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.MODEL.WEIGHTS = str(local_weights_path)
# cfg.MODEL.DEVICE = 'cpu'
assert args.numclasses > 0
cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.numclasses
assert args.thresh >= 0
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh
if args.min_size_test is not None:
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
if args.max_size_test is not None:
    cfg.INPUT.MAX_SIZE_TEST = args.max_size_test
cfg.freeze()

model = build_model(cfg)
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(str(local_weights_path))
transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
print('MIN_SIZE_TEST', cfg.INPUT.MIN_SIZE_TEST)
print('MAX_SIZE_TEST', cfg.INPUT.MAX_SIZE_TEST)
to_flip_channel = cfg.INPUT.FORMAT == 'RGB'

assert len(downloaded_dirs) == 1
local_dataset_path = downloaded_dirs[0]
imgpaths = [ p for p in local_dataset_path.glob('*') if p.suffix in IMG_EXTS ]

bs = args.bs
assert bs > 0

img_count = 0

for batch_imgpaths in batch(imgpaths, bs=bs):
    inputs = []
    raw_imgs = []
    imgname = []
    for imgpath in batch_imgpaths:
        imgname = imgpath.stem
        img = cv2.imread(str(imgpath)) 
        raw_imgs.append(img.copy())
        ih, iw = img.shape[:2]
        img = preproc(img, transform_gen)
        inputs.append({"image": img, "height": ih, "width":iw})
    
    preds = model(inputs)

    for pred, raw_img in zip(preds, raw_imgs):
        pred = pred['instances'].to('cpu')
        show_frame = raw_img.copy()
        bboxes = pred.pred_boxes.tensor.detach().numpy()
        scores = pred.scores.detach().numpy()
        pred_classes = pred.pred_classes.detach().numpy()

        for bb, score, class_ in zip(bboxes, scores, pred_classes):
            l,t,r,b = bb
            cv2.rectangle(show_frame, (int(l), int(t)), (int(r),int(b)), (255,255,0))
            cv2.putText(show_frame, '{}:{:0.2f}'.format(class_, score), (l,b), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,255,0), lineType=2)
        
        logger.report_image("Viz", f"{imgname}", iteration=img_count, image=show_frame[:,:,::-1])
        img_count += 1

logger.flush()