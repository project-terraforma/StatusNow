
import boto3
import re
from botocore import UNSIGNED
from botocore.client import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "overturemaps-us-west-2"
prefix = "release/"

response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")

releases = []
if 'CommonPrefixes' in response:
    for p in response['CommonPrefixes']:
        path = p['Prefix']
        # Extract version like 2024-03-12-alpha.0
        match = re.search(r'release/([\d\-\.\w]+)/', path)
        if match:
            releases.append(match.group(1))

releases.sort()
print("Available releases:")
for r in releases:
    print(r)

if len(releases) >= 2:
    print(f"\nTarget Releases: Current={releases[-1]}, Previous={releases[-2]}")
else:
    print("Warning: Not enough releases found.")
