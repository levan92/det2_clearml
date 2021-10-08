from pathlib import Path
import os
import argparse

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("src_bucket", help="Source bucket")
ap.add_argument("src_path", help="Source path")
ap.add_argument("expt_name", help="Experiment name")
ap.add_argument("dst_path", help="Destination path")
ap.add_argument(
    "--dst-bucket", help="Destination bucket (optional), will default to Source bucket."
)
args = ap.parse_args()

AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "https://ecs.dsta.ai")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS = os.environ.get("AWS_SECRET_ACCESS")
CERT_PATH = os.environ.get(
    "CERT_PATH", "/usr/share/ca-certificates/extra/ca.dsta.ai.crt"
)
s3_resource = boto3.resource(
    "s3",
    endpoint_url=AWS_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS,
    verify=CERT_PATH,
)

src_buck = s3_resource.Bucket(args.src_bucket)
max_step = -1
best_key = None
no_step_model_best = None
prefix = Path(args.src_path) / args.expt_name / "models"
for obj in src_buck.objects.filter(Prefix=f"{prefix}/"):
    stem = Path(obj.key).stem
    if stem == "model_best":
        no_step_model_best = obj.key
        continue
    else:
        try:
            step = int(stem.split("model_best_")[-1])
            if step > max_step:
                max_step = step
                best_key = obj.key
        except ValueError:
            print(f"Ignoring {stem}")

if best_key is None and no_step_model_best is None:
    raise Exception("No appropriate model file found.")

key_to_copy = best_key or no_step_model_best
file_name = Path(key_to_copy).name
parent_name = Path(key_to_copy).parent.parent.name

source = {"Bucket": args.src_bucket, "Key": key_to_copy}

if args.dst_bucket:
    dst_buck = s3_resource.Bucket(args.dst_bucket)
else:
    dst_buck = src_buck

dest_path = Path(args.dst_path) / parent_name / file_name
print(f'Source: {source}')
print(f'Destination path: {dest_path}')
dst_buck.copy(source, str(dest_path))
