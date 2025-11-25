"""
Weekly ML pipeline - preprocess new data and check for drift
"""

from datetime import datetime
from pathlib import Path
from prefect import flow, task

from prefect_aws import AwsCredentials
from config import BUCKET, INCOMING_PREFIX, PROCESSED_PREFIX


@task(retries=1)
def get_batch_folders():
    """Find all batch_ folders in incoming/raw"""
    aws_credentials = AwsCredentials.load("my-aws-creds")
    session = aws_credentials.get_boto3_session()
    s3 = session.client("s3")
    
    # List all objects under incoming/raw/
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=INCOMING_PREFIX,
        Delimiter='/'
    )
    
    if 'CommonPrefixes' not in response:
        return []
    
    # Filter for batch_ prefixes
    batch_folders = [
        prefix['Prefix'] for prefix in response['CommonPrefixes']
        if 'batch_' in prefix['Prefix']
    ]
    
    return batch_folders


@task(retries=1)
def get_files_from_batches(batch_folders):
    """Get all images and labels from batch folders"""
    aws_credentials = AwsCredentials.load("my-aws-creds")
    session = aws_credentials.get_boto3_session()
    s3 = session.client("s3")
    
    images = []
    labels = []
    
    for batch_folder in batch_folders:
        # Get images
        img_response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=f'{batch_folder}images/'
        )
        if 'Contents' in img_response:
            images.extend([
                obj['Key'] for obj in img_response['Contents']
                if not obj['Key'].endswith('/')
            ])
        
        # Get labels
        lbl_response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=f'{batch_folder}labels/'
        )
        if 'Contents' in lbl_response:
            labels.extend([
                obj['Key'] for obj in lbl_response['Contents']
                if not obj['Key'].endswith('/')
            ])
    
    return images, labels


@task(retries=2)
def preprocess_and_move(images, labels):
    """Preprocess images/labels and move to weekly_batch folder"""
    if not images:
        return []
    
    aws_credentials = AwsCredentials.load("my-aws-creds")
    session = aws_credentials.get_boto3_session()
    s3 = session.client("s3")
    
    # Create timestamped weekly batch folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    weekly_batch_prefix = f'{PROCESSED_PREFIX}weekly_batch_{timestamp}/'
    
    processed_files = []
    
    # Process images
    for img_key in images:
        try:
            filename = Path(img_key).name
            
            # TODO: Implement actual preprocessing
            # - Download image
            # - Resize, normalize, etc.
            # - Extract features for drift detection
            
            new_key = f'{weekly_batch_prefix}images/{filename}'
            
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={'Bucket': BUCKET, 'Key': img_key},
                Key=new_key
            )
            
            processed_files.append(new_key)
            
        except Exception as e:
            print(f"Error processing image {Path(img_key).name}: {e}")
            continue
    
    # Process labels
    for lbl_key in labels:
        try:
            filename = Path(lbl_key).name
            
            new_key = f'{weekly_batch_prefix}labels/{filename}'
            
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={'Bucket': BUCKET, 'Key': lbl_key},
                Key=new_key
            )
            
            processed_files.append(new_key)
            
        except Exception as e:
            print(f"Error processing label {Path(lbl_key).name}: {e}")
            continue
    
    return processed_files, weekly_batch_prefix


@task(retries=1)
def cleanup_batch_folders(batch_folders):
    """Delete processed batch folders from incoming/raw"""
    aws_credentials = AwsCredentials.load("my-aws-creds")
    session = aws_credentials.get_boto3_session()
    s3 = session.client("s3")
    
    for batch_folder in batch_folders:
        # List all objects in the batch folder
        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=batch_folder
        )
        
        if 'Contents' in response:
            # Delete all objects
            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
            if objects_to_delete:
                s3.delete_objects(
                    Bucket=BUCKET,
                    Delete={'Objects': objects_to_delete}
                )
    
    print(f"Cleaned up {len(batch_folders)} batch folders")


@task
def detect_drift(processed_files, override_drift):
    """Check if drift detected in new data"""
    # Placeholder - returns (drift_detected, drift_score)
    # return detect_drift({'total_images': num_processed})
    if override_drift is None:
        # Actually detect drift
        drift = (False, 0.32)
    elif override_drift is True:
        drift = (True, 0.67)
    else:
        drift = (False, 0.32)
    return drift


@task
def trigger_sagemaker_job(trigger_sagemaker, epochs):
    """Trigger sagemaker job if drift detected"""
    print(f"sagemaker triggered with {trigger_sagemaker} for {epochs} epochs")
    # aws_credentials = AwsCredentials.load("my-aws-creds")
    # session = aws_credentials.get_boto3_session()
    # sm = session.client("sagemaker")

    # # Define job name
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # training_job_name = f"weekly-drift-retrain-{timestamp}"

    # # You probably want to parameterize these; I'm giving a skeleton
    # training_params = {
    #     "TrainingJobName": training_job_name,
    #     "AlgorithmSpecification": {
    #         # e.g., use a built-in algorithm or your own container
    #         "TrainingImage": "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-training-image:latest",
    #         "TrainingInputMode": "File",
    #     },
    #     "RoleArn": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",   # change
    #     "InputDataConfig": [
    #         {
    #             "ChannelName": "training",
    #             "DataSource": {
    #                 "S3DataSource": {
    #                     "S3DataType": "S3Prefix",
    #                     "S3Uri": f"s3://{BUCKET}/{weekly_batch_prefix}images/",  # or labels if needed
    #                     "S3DataDistributionType": "FullyReplicated"
    #                 }
    #             },
    #             "ContentType": "application/x-image",  # or whatever fits your data
    #         },
    #     ],
    #     "OutputDataConfig": {
    #         "S3OutputPath": f"s3://{BUCKET}/models/{training_job_name}"
    #     },
    #     "ResourceConfig": {
    #         "InstanceType": "ml.m5.xlarge",  # choose suitable instance
    #         "InstanceCount": 1,
    #         "VolumeSizeInGB": 50
    #     },
    #     "StoppingCondition": {
    #         "MaxRuntimeInSeconds": 3600 * 24  # 1 day, adjust as needed
    #     },
    #     "HyperParameters": {
    #         "epochs": "50",
    #         # add more as needed
    #     },
    #     # optionally tags, metric definitions, etc.
    # }

    # response = sm.create_training_job(**training_params)
    # job_arn = response["TrainingJobArn"]
    # print(f"Started SageMaker training job, ARN: {job_arn}")



@flow(log_prints=True)
def weekly_ingestion_pipeline(override_drift: bool | None = None):
    """Check for new data, preprocess, detect drift, retrain if needed"""
    
    # Find all batch folders
    batch_folders = get_batch_folders()
    
    if not batch_folders:
        print("No batch folders to process")
        return {"status": "no_new_data"}
    
    print(f"Found {len(batch_folders)} batch folders")
    
    # Get all images and labels from batches
    images, labels = get_files_from_batches(batch_folders)
    print(f"Found {len(images)} images and {len(labels)} labels")
    
    if not images:
        print("No images found in batch folders")
        return {"status": "no_new_data"}
    
    # Preprocess and move to weekly_batch folder
    processed_files, weekly_batch_prefix = preprocess_and_move(images, labels)
    print(f"Processed {len(processed_files)} files to {weekly_batch_prefix}")
    
    # Clean up original batch folders
    cleanup_batch_folders(batch_folders)
    
    # Check drift with actual processed data
    drift_detected, drift_score = detect_drift(processed_files, override_drift)
    print(f"Drift score: {drift_score:.4f}")
    
    if drift_detected:
        print("Drift detected - triggering retraining")
        training_result = trigger_sagemaker_job(
            trigger_reason="drift_detected",
            epochs=50
        )
        
        return {
            "status": "retrained",
            "drift_score": drift_score,
            "processed": len(images),
            "weekly_batch": weekly_batch_prefix,
            "training_result": training_result
        }
    else:
        print("No significant drift")
        return {
            "status": "no_retrain_needed",
            "drift_score": drift_score,
            "processed": len(images),
            "weekly_batch": weekly_batch_prefix
        }


if __name__ == "__main__":
    weekly_ingestion_pipeline()