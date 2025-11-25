"""
Single entry point for all Prefect flows using .serve()
Run with: python run_all.py

This will start all flows with their schedules in a single process.
"""
from prefect import serve
# from flows.daily_upload import daily_batch_upload
# from flows.weekly_pipeline import weekly_ml_pipeline
# from flows.training_pipeline import sagemaker_training_pipeline
from prefect import flow
from pathlib import Path


@flow(name="test1", log_prints=True)
def test1():
    print("Test1")
    return {
        "status": "okay"
    }

@flow(name="test2", log_prints=True)
def test2():
    print("gonna update some stuff")
    print("TEST2")
    return {
        "status": "fail"
    }

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
    
    deployment = test2.from_source(
        source="https://github.com/mscac-csc2701-fa25/prefect_flows.git",
        entrypoint="flows/run_all_2.py:test2"
    ).deploy(
        name="test_2",
        work_pool_name="my-ec2-process-pool",
        cron="0 1 * * *"
    )

    # other flows
    # weekly = weekly_ml_pipeline.deploy(...)
    # train = sagemaker_training_pipeline.deploy(...)

    print("Deployments created:", deployment)  


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down flows...")
        print("="*60 + "\n")