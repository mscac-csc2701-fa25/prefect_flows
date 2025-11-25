"""
Weekly ML pipeline - preprocess new data and check for drift
"""

import boto3
from datetime import datetime
from pathlib import Path
from prefect import flow, task

from prefect_flows.flows.training_pipeline import sagemaker_training_pipeline
from config import BUCKET, INCOMING_PREFIX, PROCESSED_PREFIX


@task(retries=1)
def get_raw_files():
    """Check for files in incoming/raw"""
    s3 = boto3.client('s3')
    
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=INCOMING_PREFIX
    )
    
    if 'Contents' not in response:
        return []
    
    files = [obj['Key'] for obj in response['Contents'] 
             if not obj['Key'].endswith('/')]
    
    return files


@task(retries=2)
def preprocess_and_move(file_keys):
    """Preprocess images and move to processed folder"""
    if not file_keys:
        return []
    
    s3 = boto3.client('s3')
    processed_files = []
    
    for file_key in file_keys:
        try:
            filename = Path(file_key).name
            
            # TODO: Implement actual preprocessing
            # - Download image
            # - Resize, normalize, etc.
            # - Extract features for drift detection
            
            new_key = f'{PROCESSED_PREFIX}{filename}'
            
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={'Bucket': BUCKET, 'Key': file_key},
                Key=new_key
            )
            s3.delete_object(Bucket=BUCKET, Key=file_key)
            
            processed_files.append(new_key)
            
        except Exception as e:
            print(f"Error processing {Path(file_key).name}: {e}")
            continue
    
    return processed_files


@task
def detect_drift(processed_files):
    """Check if drift detected in new data"""
    # Placeholder - returns (drift_detected, drift_score)
    # return detect_drift({'total_images': num_processed})
    return (False, 0.3)


@flow(log_prints=True)
def weekly_ml_pipeline():
    """Check for new data, preprocess, detect drift, retrain if needed"""
    
    # Check for raw files
    raw_files = get_raw_files()
    
    if not raw_files:
        print("No new data to process")
        return {"status": "no_new_data"}
    
    print(f"Found {len(raw_files)} files to process")
    
    # Preprocess and move to processed
    processed_files = preprocess_and_move(raw_files)
    print(f"Processed {len(processed_files)} files")
    
    # Check drift with actual processed data
    drift_detected, drift_score = detect_drift(processed_files)
    print(f"Drift score: {drift_score:.4f}")
    
    if drift_detected:
        print("Drift detected - triggering retraining")
        training_result = sagemaker_training_pipeline(
            trigger_reason="drift_detected",
            epochs=50
        )
        
        return {
            "status": "retrained",
            "drift_score": drift_score,
            "processed": len(processed_files),
            "training_result": training_result
        }
    else:
        print("No significant drift")
        return {
            "status": "no_retrain_needed",
            "drift_score": drift_score,
            "processed": len(processed_files)
        }


if __name__ == "__main__":
    weekly_ml_pipeline()