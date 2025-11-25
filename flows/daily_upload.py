"""
Daily batch upload - moves images from simulation_pool to incoming
"""

import boto3
from datetime import datetime
from pathlib import Path
from prefect import task, flow

from config import BATCH_SIZE
from config import BUCKET


@task(retries=1)
def get_available_images(limit=BATCH_SIZE):
    """Get next batch of images from simulation pool"""
    s3 = boto3.client('s3')
    
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix='datasets/simulation_pool/images/',
        MaxKeys=limit * 2
    )
    
    if 'Contents' not in response:
        return []
    
    images = []
    for obj in response['Contents']:
        key = obj['Key']
        if not key.endswith('/') and key.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(key)
            if len(images) >= limit:
                break
    
    return images


@task(retries=2)
def move_batch(image_keys):
    """Move images and labels to incoming"""
    if not image_keys:
        return 0
    
    s3 = boto3.client('s3')
    batch_date = datetime.now().strftime("%Y%m%d")
    moved = 0
    
    for image_key in image_keys:
        try:
            image_name = Path(image_key).name
            image_stem = Path(image_key).stem
            
            new_image_key = f'datasets/incoming/raw/batch_{batch_date}/images/{image_name}'
            new_label_key = f'datasets/incoming/raw/batch_{batch_date}/labels/{image_stem}.txt'
            
            # Copy and delete image
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={'Bucket': BUCKET, 'Key': image_key},
                Key=new_image_key
            )
            s3.delete_object(Bucket=BUCKET, Key=image_key)
            print(f"- Moved image: {image_name}")

            # Try to move label
            label_key = image_key.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            try:
                s3.copy_object(
                    Bucket=BUCKET,
                    CopySource={'Bucket': BUCKET, 'Key': label_key},
                    Key=new_label_key
                )
                s3.delete_object(Bucket=BUCKET, Key=label_key)
                print(f"- Moved label: {image_stem}.txt")
            except s3.exceptions.NoSuchKey:
                print("- (No label found)")
                pass
            
            moved += 1
            
        except Exception as e:
            print(f"Error moving {Path(image_key).name}: {e}")
            continue
    
    return moved


@task
def get_stats():
    """Get counts from pool and incoming"""
    s3 = boto3.client('s3')
    
    # Count remaining in pool
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix='datasets/simulation_pool/images/'
        )
        pool_count = len([obj for obj in response.get('Contents', []) 
                         if not obj['Key'].endswith('/')])
    except:
        pool_count = 0
    
    # Count in incoming
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix='datasets/incoming/raw/'
        )
        incoming_count = len([obj for obj in response.get('Contents', []) 
                             if not obj['Key'].endswith('/') and 'images/' in obj['Key']])
    except:
        incoming_count = 0
    
    return pool_count, incoming_count


@flow(log_prints=True)
def daily_batch_upload():
    """Move batch of images from pool to incoming"""
    print("\n" + "="*60)
    print(f"DAILY BATCH UPLOAD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    images = get_available_images(BATCH_SIZE)
    
    if not images:
        print("No images available")
        return {"status": "no_images", "moved": 0}
    
    moved = move_batch(images)
    pool_remaining, incoming_total = get_stats()
    

    print("\n" + "="*60)
    print("BATCH COMPLETE!")
    print(f"   Moved: {moved} images")
    print(f"   Remaining in pool: {pool_remaining}")
    print(f"   Total in incoming: {incoming_total}")
    print("="*60 + "\n")
    return {
        "status": "success",
        "moved": moved,
        "pool_remaining": pool_remaining,
        "incoming_total": incoming_total
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'serve':
        daily_batch_upload.serve(
            name="daily-batch-upload",
            cron="0 1 * * *"
        )
    else:
        daily_batch_upload()