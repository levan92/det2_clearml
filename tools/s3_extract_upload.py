from clearml import Task

import os
import zipfile, tarfile
from io import BytesIO
from pathlib import Path
import argparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config

GB = 1024 ** 3
NO_MPUPLOAD = TransferConfig(multipart_threshold=20*GB)

def buffered_read(stream_body, chunksize=1024**3):
    byte_arr = bytearray()
    for i, chunk in enumerate(stream_body.iter_chunks(chunk_size=chunksize)):
        print(f'chunk {i}')
        print(len(chunk))
        byte_arr.extend(chunk)
        print(len(byte_arr))
    return byte_arr

def extract_upload(s3_resource, obj, dest_bucket, upload_dir_path, verbose=False, filetype='zip'):
    upload_dir_path = Path(upload_dir_path)
    if filetype == 'zip':
        with BytesIO() as buffer:
            print(f"Reading {filetype} file..")
            obj.download_fileobj(buffer)
            print("Iterating through file")
            z = zipfile.ZipFile(buffer)
            for filename in z.namelist():
                file_info = z.getinfo(filename)
                if file_info.is_dir():
                    if verbose:
                        print(f"Skipping {filename} as it is dir.")
                    continue

                upload_path = upload_dir_path / filename
                if verbose:
                    print(filename)
                    print("Uploading to", upload_path)
                s3_resource.meta.client.upload_fileobj(
                    z.open(filename), Bucket=dest_bucket, Key=f"{upload_path}", Config=NO_MPUPLOAD
                )
    elif filetype =='tar':
        with BytesIO(buffered_read(obj.get()['Body'])) as buffer:
            with tarfile.open(fileobj=buffer, mode='r') as tar:
                for tarinfo in tar:
                    fname = tarinfo.name
                    if not tarinfo.isfile():
                        continue 
                    if fname is None:
                        continue
                    upload_path = upload_dir_path / fname
                    if verbose:
                        print(fname)
                        print("Uploading to", upload_path)

                    s3_resource.meta.client.upload_fileobj(
                        tar.extractfile(tarinfo), Bucket=dest_bucket, Key=f"{upload_path}", Config=NO_MPUPLOAD
                    )


ap = argparse.ArgumentParser()
ap.add_argument("src_bucket", help="Source bucket")
ap.add_argument("src_path", help="Source path")
ap.add_argument("dst_path", help="Destination path")
ap.add_argument(
    "--dst-bucket", help="Destination bucket (optional), will default to Source bucket."
)
ap.add_argument(
    "--verbose",
    help="print out current upload filename as it progresses",
    action="store_true",
)
ap.add_argument("--remote", help="use clearml to remotely run job", action="store_true")
args = ap.parse_args()

AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "https://ecs.dsta.ai")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS = os.environ.get("AWS_SECRET_ACCESS")
CERT_PATH = os.environ.get(
    "CERT_PATH", "/usr/share/ca-certificates/extra/ca.dsta.ai.crt"
)
CERT_PATH = CERT_PATH if CERT_PATH else None

if args.remote:
    """
    clearml task init
    """
    cl_task = Task.init(
        project_name="lfn",
        task_name="co3d_extract_upload",
        task_type="data_processing",
        output_uri="s3://ecs.dsta.ai:80/clearml-models/default/",
    )
    cl_task.set_base_docker(
        f"harbor.dsta.ai/public/lfn_python38:v2 --env GIT_SSL_NO_VERIFY=true --env AWS_ACCESS_KEY={AWS_ACCESS_KEY} --env AWS_SECRET_ACCESS={AWS_SECRET_ACCESS}"
    )
    cl_task.execute_remotely(queue_name="lfn-queue-1xV100-128ram", exit_process=True)


src_buck = args.src_bucket
if args.dst_bucket:
    dst_buck = args.dst_bucket
else:
    dst_buck = src_buck
upload_folder = Path(args.dst_path)

s3_resource = boto3.resource(
    "s3",
    endpoint_url=AWS_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS,
    verify=CERT_PATH,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

src_bucket = s3_resource.Bucket(src_buck)

src_path = args.src_path
if not src_path.endswith("/"):
    src_path = src_path + "/"

for obj in src_bucket.objects.filter(Prefix=f"{src_path}"):
    if obj.key.endswith(".zip"):
        filetype = 'zip'
    elif obj.key.endswith('.tar'):
        filetype = 'tar'
    else: 
        filetype = None

    if filetype is not None:
        print("Extracting and uploading: ", obj.key)
        extract_upload(
            s3_resource, obj.Object(), dst_buck, upload_folder, verbose=args.verbose, filetype=filetype
        )
