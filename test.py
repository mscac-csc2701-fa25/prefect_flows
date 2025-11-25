"""
Daily Batch Upload - Moves 10 images per day from simulation_pool to incoming
Run with: python batch_upload.py serve
"""

import boto3
from datetime import datetime
from pathlib import Path
from prefect import task, flow

# Configuration
BUCKET = 'smoke-fire-detection-bucket'
BATCH_SIZE = 5


@task(name="get-batch-images", retries=1)
def get_available_images(limit=BATCH_SIZE):
    """Get the next batch of images from simulation pool"""
    print(f"🔍 Looking for {limit} images in simulation pool...")
    s3 = boto3.client('s3')
    
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix='datasets/simulation_pool/images/',
        MaxKeys=limit * 2  # Get extra in case some are folders
    )
    
    if 'Contents' not in response:
        print("❌ No images found in simulation_pool/images/")
        return []
    
    # Filter out folders, get actual image files
    images = []
    for obj in response['Contents']:
        key = obj['Key']
        if not key.endswith('/') and any(key.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            images.append(key)
            if len(images) >= limit:
                break
    
    print(f"  ✅ Found {len(images)} images to move")
    return images


@task(name="move-batch", retries=2)
def move_batch_to_incoming(image_keys):
    """Move a batch of images (and labels) to incoming with datestamp"""
    if not image_keys:
        print("⚠️  No images to move")
        return 0
    
    s3 = boto3.client('s3')
    batch_date = datetime.now().strftime("%Y%m%d")
    moved_count = 0
    
    print(f"\n📦 Moving batch to incoming/batch_{batch_date}/")
    
    for image_key in image_keys:
        try:
            image_name = Path(image_key).name
            image_stem = Path(image_key).stem
            
            # New destinations with batch folder
            new_image_key = f'datasets/incoming/batch_{batch_date}/images/{image_name}'
            new_label_key = f'datasets/incoming/batch_{batch_date}/labels/{image_stem}.txt'
            
            # Move image
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={'Bucket': BUCKET, 'Key': image_key},
                Key=new_image_key
            )
            s3.delete_object(Bucket=BUCKET, Key=image_key)
            print(f"  ✓ Moved image: {image_name}")
            
            # Move label (if it exists)
            label_key = image_key.replace('/images/', '/labels/')
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                label_key = label_key.replace(ext, '.txt')
            
            try:
                s3.copy_object(
                    Bucket=BUCKET,
                    CopySource={'Bucket': BUCKET, 'Key': label_key},
                    Key=new_label_key
                )
                s3.delete_object(Bucket=BUCKET, Key=label_key)
                print(f"    └─ Label: {image_stem}.txt")
            except s3.exceptions.NoSuchKey:
                print(f"    └─ (no label found)")
            
            moved_count += 1
            
        except Exception as e:
            print(f"  ❌ Error moving {Path(image_key).name}: {e}")
            continue
    
    return moved_count


@task(name="count-stats")
def get_stats():
    """Get counts from both folders"""
    s3 = boto3.client('s3')
    
    # Count remaining in simulation pool
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
            Prefix='datasets/incoming/'
        )
        incoming_count = len([obj for obj in response.get('Contents', []) 
                             if not obj['Key'].endswith('/') and 'images/' in obj['Key']])
    except:
        incoming_count = 0
    
    return pool_count, incoming_count


@flow(name="daily-batch-upload", log_prints=True)
def daily_batch_upload():
    """Main flow - move 10 images daily"""
    print("\n" + "="*60)
    print(f"🚀 DAILY BATCH UPLOAD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Get batch of images
    images = get_available_images(BATCH_SIZE)
    
    if not images:
        print("\n❌ No images available in simulation pool!")
        return {"status": "no_images", "moved": 0}
    
    # Move batch
    moved = move_batch_to_incoming(images)
    
    # Get stats
    pool_remaining, incoming_total = get_stats()
    
    print("\n" + "="*60)
    print("✅ BATCH COMPLETE!")
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
    
    print("\n" + "="*60)
    print("📦 DAILY BATCH UPLOAD SYSTEM")
    print("="*60)
    print(f"Bucket: {BUCKET}")
    print(f"Batch size: {BATCH_SIZE} images/day")
    print("="*60 + "\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'serve':
        print("🚀 Starting Prefect deployment...")
        print("   Schedule: Daily at 1:00 AM")
        print("   Press Ctrl+C to stop\n")
        
        # Daily at 1 AM
        daily_batch_upload.serve(
            name="daily-batch-upload",
            cron="0 1 * * *"  # Every day at 1:00 AM
        )
        
    else:
        print("🚀 Single run mode - moving one batch now")
        print("   Tip: Use 'python batch_upload.py serve' for daily schedule\n")
        
        # Run once immediately
        daily_batch_upload()