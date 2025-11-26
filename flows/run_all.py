"""
Single entry point for all Prefect flows using .serve()
Run with: python run_all.py

This will start all flows with their schedules in a single process.
"""
from prefect import serve
from flows.daily_upload import daily_batch_upload
from flows.weekly_data_ingest_and_drift import weekly_ingestion_pipeline
from flows.evaluate_pipeline import evaluate_pipeline
# from flows.training_pipeline import sagemaker_training_pipeline


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
    
    # Deployment 1
    # deployment = daily_batch_upload.from_source(
    #     source="https://github.com/mscac-csc2701-fa25/prefect_flows.git",
    #     entrypoint="flows/daily_upload.py:daily_batch_upload"
    # ).deploy(
    #     name="daily_batch_upload",
    #     work_pool_name="my-ec2-process-pool",
    #     cron="0 1 * * *",
    #     job_variables={"pip_packages": ["prefect-aws"]}
    # )

    # print("Deployments created:", deployment)  
    
    # Deployment 2
    deployment = weekly_ingestion_pipeline.from_source(
        source="https://github.com/mscac-csc2701-fa25/prefect_flows.git",
        entrypoint="flows/weekly_data_ingest_and_drift.py:weekly_ingestion_pipeline"
    ).deploy(
        name="weekly_ingestion_pipeline",
        work_pool_name="my-ec2-process-pool",
        cron="0 2 * * 0",  # Sundays at 2 AM UTC
        job_variables={"pip_packages": ["prefect-aws", "scipy", "pillow", "numpy"]}
    )

    print("Deployments created:", deployment)  

    # Deployment 3
    # deployment = evaluate_pipeline.from_source(
    #     source="https://github.com/mscac-csc2701-fa25/prefect_flows.git",
    #     entrypoint="flows/evaluate_pipeline.py:evaluate_pipeline"
    # ).deploy(
    #     name="evaluate_pipeline",
    #     work_pool_name="my-ec2-process-pool",
    #     job_variables={"pip_packages": ["mlflow"]}
    # )

    # print("Deployments created:", deployment)  

    # Serve all flows together
    # serve(

        # Daily upload flow - runs at 1 AM every day
        # daily_batch_upload.to_deployment(
        #     name="daily-batch-upload",
        #     cron="0 1 * * *",  # Daily at 1 AM UTC
        #     tags=["production", "s3", "daily"]
        # ),
        
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
    # )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down flows...")
        print("="*60 + "\n")