import sagemaker
import json
import boto3
import user_keys
import utils
import os
import sys
import argparse
import time
import training_parameters

if __name__ == '__main__':
    # TODO: Logging might also be good

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name_prefix")
    parser.add_argument("--data_upload")
    parser.add_argument("--s3_location")
    parser.add_argument("--training")
    parser.add_argument("--deploy")

    args = parser.parse_args()

    bucket_prefix, data_upload, s3_output_location, training, deploy = utils.argument_parser(
        args)

    # setting up variables
    session = boto3.session.Session()
    sagemaker_session = sagemaker.Session()
    role = user_keys.arn_role

    # setting up resources
    s3_resource = boto3.resource('s3')

    # create bucket to store the data

    bucket_name, bucket_response = utils.create_bucket(
        bucket_prefix=bucket_prefix,
        s3_connection=s3_resource,
        session=session
    )

    if data_upload == 1:
        prefix = 'demo/supervised'
        train_channel = prefix + '/train'
        validation_channel = prefix + '/validation'

        s3_train_data, time_taken = utils.upload_data_to_bucket(
            data_path="./data/dbpedia.train",
            bucket_name=bucket_name,
            channel=train_channel,
            session=sagemaker_session
        )
        print("UPLOAD SUCCESS")
        print("Path: {}\nTime: {}".format(s3_train_data, time_taken))

        s3_validation_data, time_taken = utils.upload_data_to_bucket(
            data_path="./data/dbpedia.validation",
            bucket_name=bucket_name,
            channel=validation_channel,
            session=sagemaker_session
        )
        print("UPLOAD SUCCESS")
        print("Path: {}\nTime: {}".format(s3_validation_data, time_taken))

        s3_output_location = 's3://{}/{}/output'.format(bucket_name, prefix)
        print("DATA UPLOAD COMPLETE")

    print("S3 LOCATION: {}".format(s3_output_location))

    if training == 1:
        print("TRAINING STARTING")
        region_name = session.region_name
        container = sagemaker.image_uris.retrieve(
            'blazingtext', region_name, 'latest')
        print('Using SageMaker BlazingText container: {} ({})'.format(
            container, region_name))

        bt_model = sagemaker.estimator.Estimator(
            image_uri=container,
            role=role,
            instance_count=training_parameters.train_instance_count,
            instance_type=training_parameters.train_instance_type,
            volume_size=training_parameters.train_volume_size,
            max_run=training_parameters.train_max_run,
            input_mode=training_parameters.input_mode,
            output_path=s3_output_location,
            sagemaker_session=sagemaker_session
        )

        bt_model.set_hyperparameters(
            mode=training_parameters.mode,
            epochs=training_parameters.epochs,
            min_count=training_parameters.min_count,
            learning_rate=training_parameters.learning_rate,
            vector_dim=training_parameters.vector_dim,
            early_stopping=training_parameters.early_stopping,
            patience=training_parameters.patience,
            min_epochs=training_parameters.min_epochs,
            word_ngrams=training_parameters.word_ngrams
        )

        train_data = sagemaker.inputs.TrainingInput(
            s3_train_data,
            distribution='FullyReplicated',
            content_type='text/plain',
            s3_data_type='S3Prefix'
        )

        validation_data = sagemaker.inputs.TrainingInput(
            s3_validation_data,
            distribution='FullyReplicated',
            content_type='text/plain',
            s3_data_type='S3Prefix'
        )

        data_channels = {
            'train': train_data,
            'validation': validation_data
        }

        print (data_channels)

        bt_model.fit(
            inputs=data_channels,
            logs=True
        )

    if deploy == 1:
        text_classifier = bt_model.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge'
        )
