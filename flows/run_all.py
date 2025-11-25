"""
Single entry point for all Prefect flows using .serve()
Run with: python run_all.py

This will start all flows with their schedules in a single process.
"""
from prefect import serve
from prefect_flows.flows.daily_upload import daily_batch_upload
# from prefect_flows.flows.weekly_pipeline import weekly_ml_pipeline
# from prefect_flows.flows.training_pipeline import sagemaker_training_pipeline



def main():
    """Start all flows with their schedules"""
    
    print("\n" + "="*60)
    print("STARTING PREFECT FLOWS")
    print("="*60)
    print("\nFlows:")
    print("  1. Daily Batch Upload    - Every day at 1:00 AM UTC")
    print("  2. Weekly ML Pipeline    - Every Sunday at 2:00 AM UTC")
    print("  3. Training Pipeline     - Manual trigger only")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop all flows")
    print("="*60 + "\n")
    
    # Serve all flows together
    serve(

        # Daily upload flow - runs at 1 AM every day
        daily_batch_upload.to_deployment(
            name="daily-batch-upload",
            cron="0 1 * * *",  # Daily at 1 AM UTC
            tags=["production", "s3", "daily"]
        ),
        
        # # Weekly ML pipeline - runs at 2 AM every Sunday
        # weekly_ml_pipeline.to_deployment(
        #     name="weekly-ml-pipeline",
        #     cron="0 2 * * 0",  # Sundays at 2 AM UTC
        #     tags=["production", "ml", "weekly"]
        # ),
        
        # # Training pipeline - no schedule (manual only)
        # # Can be triggered by weekly_ml_pipeline when drift is detected
        # sagemaker_training_pipeline.to_deployment(
        #     name="sagemaker-training-manual",
        #     tags=["production", "training", "sagemaker"]
        # ),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down flows...")
        print("="*60 + "\n")