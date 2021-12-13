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
NO_MPUPLOAD = TransferConfig(multipart_threshold=20 * GB)


def buffered_read(stream_body, chunksize=1 * GB):
    byte_arr = bytearray()
    for i, chunk in enumerate(stream_body.iter_chunks(chunk_size=chunksize)):
        byte_arr.extend(chunk)
        print(f"Downloaded {len(byte_arr)//GB}GB so far (chunk {i})..")
    return byte_arr


def extract_upload(
    s3_resource, obj, dest_bucket, upload_dir_path, verbose=False, filetype="zip"
):
    upload_dir_path = Path(upload_dir_path)
    if filetype == "zip":
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
                    z.open(filename),
                    Bucket=dest_bucket,
                    Key=f"{upload_path}",
                    Config=NO_MPUPLOAD,
                )
    elif filetype == "tar":
        with BytesIO(buffered_read(obj.get()["Body"])) as buffer:
            with tarfile.open(fileobj=buffer, mode="r") as tar:
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
                        tar.extractfile(tarinfo),
                        Bucket=dest_bucket,
                        Key=f"{upload_path}",
                        Config=NO_MPUPLOAD,
                    )


ap = argparse.ArgumentParser()
ap.add_argument("src_bucket", help="Source bucket")
ap.add_argument("src_path", help="Source path")
ap.add_argument("dst_path", help="Destination path")
ap.add_argument(
    "--src-is-dir",
    help="Flag to indicate that given src path is a directory. Will iteratively extract any files in it ending with .zip or .tar.",
    action="store_true",
)
ap.add_argument(
    "--dst-bucket", help="Destination bucket (optional), will default to Source bucket."
)
ap.add_argument(
    "--verbose",
    help="print out current upload filename as it progresses",
    action="store_true",
)
ap.add_argument("--remote", help="use clearml to remotely run job", action="store_true")
ap.add_argument("--clml-proj", default="COCO", help="ClearML Project Name")
ap.add_argument("--clml-task-name", default="extract_upload", help="ClearML Task Name")
ap.add_argument(
    "--clml-task-type",
    default="data_processing",
    help="ClearML Task Type, e.g. training, testing, inference, etc",
    choices=[
        "training",
        "testing",
        "inference",
        "data_processing",
        "application",
        "monitor",
        "controller",
        "optimizer",
        "service",
        "qc",
        "custom",
    ],
)
ap.add_argument(
    "--docker-img",
    default="harbor.dsta.ai/public/detectron2:best-val-steps-nosavepred",
    help="Base docker image to pull",
)
ap.add_argument("--queue", default="queue-1xV100-64ram", help="ClearML Queue")
args = ap.parse_args()

AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
CERT_PATH = os.environ.get("CERT_PATH")
CERT_DL_URL = os.environ.get("CERT_DL_URL")

CERT_PATH = CERT_PATH if CERT_PATH else None
if CERT_PATH:
    if not os.path.exists(CERT_PATH) and CERT_DL_URL:
        import wget
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        print(f"Downloading cert from {CERT_DL_URL}")
        wget.download(CERT_DL_URL)
        CERT_PATH = Path(CERT_DL_URL).name
    assert os.path.exists(CERT_PATH)

if args.remote:
    """
    clearml task init
    """
    cl_task = Task.init(
        project_name=args.clml_proj,
        task_name=args.clml_task_name,
        task_type=args.clml_task_type,
        output_uri="s3://ecs.dsta.ai:80/clearml-models/default/",
    )
    cl_task.set_base_docker(
        f"{args.docker_img} --env GIT_SSL_NO_VERIFY=true --env AWS_ENDPOINT_URL={AWS_ENDPOINT_URL} --env AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID} --env AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY} --env CERT_PATH={CERT_PATH} --env CERT_DL_URL={CERT_DL_URL}"
    )
    cl_task.execute_remotely(queue_name=args.queue, exit_process=True)


src_buck = args.src_bucket
if args.dst_bucket:
    dst_buck = args.dst_bucket
else:
    dst_buck = src_buck
upload_folder = Path(args.dst_path)

print(f"Connecting to {AWS_ENDPOINT_URL}")
s3_resource = boto3.resource(
    "s3",
    endpoint_url=AWS_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    verify=CERT_PATH,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

src_bucket = s3_resource.Bucket(src_buck)

src_path = args.src_path
if args.src_is_dir and not src_path.endswith("/"):
    src_path = src_path + "/"

print(f"Looking for objects with prefix: {src_path}")

for obj in src_bucket.objects.filter(Prefix=f"{src_path}"):
    if obj.key.endswith(".zip"):
        filetype = "zip"
    elif obj.key.endswith(".tar"):
        filetype = "tar"
    else:
        filetype = None

    if filetype is not None:
        print("Extracting and uploading: ", obj.key)
        extract_upload(
            s3_resource,
            obj.Object(),
            dst_buck,
            upload_folder,
            verbose=args.verbose,
            filetype=filetype,
        )
