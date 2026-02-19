
import boto3
from botocore import UNSIGNED
from botocore.client import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "overturemaps-us-west-2"
release = "2026-02-18.0"
prefix = f"release/{release}/theme=places/type=place/"

print(f"Listing objects in {bucket}/{prefix} ...")
response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)

if 'Contents' in response:
    for obj in response['Contents']:
        size_mb = obj['Size'] / (1024 * 1024)
        print(f"File: {obj['Key']}, Size: {size_mb:.2f} MB")
else:
    print("No files found.")
