import uuid
import boto3
import time

def create_unique_name(resource_name = 'kaambhari'):
    return ''.join([resource_name, "-", str(uuid.uuid4())])

def create_bucket(bucket_prefix = 'kaambhari', s3_connection = None, session = None):
    if not session:
        curr_region = boto3.session.Session().region_name
    else:
        curr_region = session.region_name

    bucket_name = create_unique_name(bucket_prefix)
    bucket_response = s3_connection.create_bucket(
        Bucket = bucket_name,
        CreateBucketConfiguration = {
            'LocationConstraint': curr_region
        }
    )

    return bucket_name, bucket_response

def upload_data_to_bucket(data_path, bucket_name, channel, session):
    try:
        start = time.time()
        session.upload_data(
            path = data_path,
            bucket = bucket_name,
            key_prefix = channel
        )
        end = time.time()
    except Exception as e:
        return (e, -1)
    
    return ('s3://{}/{}'.format(bucket_name, channel), end-start)

def argument_parser(args):
    if not str(args.bucket_name_prefix):
        bucket_prefix = 'kaambhari-sagemaker'
    else:
        bucket_prefix = str(args.bucket_name_prefix)

    if not args.data_upload:
        data_upload = 0
    else:
        data_upload = int(args.data_upload)

    if not args.s3_location:
        s3_output_location = ''
    else:
        s3_output_location = args.s3_location

    if not args.training:
        training = 0
    else:
        training = int(args.training)

    if not args.deploy:
        deploy = 0
    else:
        deploy = int(args.deploy)

    return (bucket_prefix, data_upload, s3_output_location, training, deploy)



