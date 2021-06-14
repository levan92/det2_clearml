import os
from pathlib import Path
import tarfile

def download_dir_from_s3(s3_resource, bucket_name, remote_dir_name, local_dir, untar=True):
    buck = s3_resource.Bucket(bucket_name)
    for obj in buck.objects.filter(Prefix=str(remote_dir_name)):
        remote_rel_path = Path(obj.key).relative_to(remote_dir_name)
        local_fp = local_dir / remote_rel_path
        local_fp.parent.mkdir(parents=True, exist_ok=True)
        if not local_fp.is_file():
            print(f'Downloading {obj.key} from S3..')
            buck.download_file(obj.key, str(local_fp))
            if local_fp.suffix in ['.tar','.tar.gz','.tgz']:
                print('Untarring..')
                tar = tarfile.open(local_fp)
                tar.extractall(local_dir)
                tar.close()


def upload_dir_to_s3(s3_resource, bucket_name, local_dir, remote_dir):
    buck = s3_resource.Bucket(bucket_name)

    for root,dirs,files in os.walk(str(local_dir)):
        for file in files: 
            local_fp = Path(root)/file
            rel_path = Path(root).relative_to(local_dir)
            remote_fp = Path(remote_dir)/ rel_path / file
            print(f'Uploading {local_fp} to S3 {remote_dir}')
            buck.upload_file(str(local_fp), str(remote_fp))
