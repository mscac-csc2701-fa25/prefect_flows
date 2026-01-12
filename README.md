# Prefect Workflow Orchestrator for MLOPs

This project implements a **MLOps pipeline** using **Prefect** to orchestrate workflows, including data drift detection and retraining pipelines.


## Architecture

![AWS Cloud Architecture](https://github.com/user-attachments/assets/36a1c8c2-90bd-4f28-91ce-c98bc21dbaf3)

## Resources
▶️ [Video](https://drive.google.com/file/d/1EO57USC1840ea61tjymcoT-Xkpl7xF6P/view)

📊 [Project Presentation Slides](https://docs.google.com/presentation/d/1UmJI0nPxKvHcp8lfnWX6v3QKNhzeMd30BHC5Jz9sE00/edit?slide=id.g4dfce81f19_0_45#slide=id.g4dfce81f19_0_45)

## Tech Stack:

Experiment Tracking & Model Registry
- [MLflow](https://mlflow.org/) – Track experiments, log metrics, and manage model registry

Workflow Orchestration & Monitoring
- [Prefect](https://www.prefect.io/) – Define, schedule, and monitor pipelines with ease

AWS Services
- **S3** – Store datasets, MLflow artifacts, and SageMaker scripts
- **SageMaker** – Compute platform for retraining models
- **EventBridge + Lambda** – Event-driven triggers for automated retraining
- **EC2** – Host the MLflow Tracking Server


### Lambda Function
> Sidenote: This AWS Lambda function is triggered by SageMaker events and programmatically starts a Prefect flow for model evaluation.
``` python

import os
import json
import urllib.request

PREFECT_ACCOUNT_ID = os.environ['PREFECT_ACCOUNT_ID']
PREFECT_WORKSPACE_ID = os.environ['PREFECT_WORKSPACE_ID']
PREFECT_DEPLOYMENT_ID = os.environ['PREFECT_DEPLOYMENT_ID']
PREFECT_API_KEY = os.environ['PREFECT_API_KEY']

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))
    
    training_job_name = event.get('detail', {}).get('TrainingJobName', 'test-training-job')
    s3_model_path = event.get('detail', {}).get('ModelArtifacts', {}).get('S3ModelArtifacts', '')

    print("Training job name:", training_job_name)
    print("S3 model path:", s3_model_path)

    payload = {
        "name": f"sagemaker_model_eval_{training_job_name}",
        "tags": ["sagemaker", "model_evaluation"],
        "parameters": {
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.prefect.cloud/api/accounts/{PREFECT_ACCOUNT_ID}/workspaces/{PREFECT_WORKSPACE_ID}/deployments/{PREFECT_DEPLOYMENT_ID}/create_flow_run",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PREFECT_API_KEY}"
        }
    )
    print("url", f"https://api.prefect.cloud/api/accounts/{PREFECT_ACCOUNT_ID}/workspaces/{PREFECT_WORKSPACE_ID}/deployments/{PREFECT_DEPLOYMENT_ID}/create_flow_run")
    try:
        with urllib.request.urlopen(req) as response:
            result = response.read()
            print("Prefect flow trigger response:", result.decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print("Error triggering Prefect flow:", e)
        print("Error response body:", error_body)  # This will show the actual validation error
        return {
            "statusCode": 500,
            "body": json.dumps(f"Failed to trigger Prefect flow: {e} - {error_body}")
        }
    except Exception as e:
        print("Error triggering Prefect flow:", e)
        return {
            "statusCode": 500,
            "body": json.dumps(f"Failed to trigger Prefect flow: {e}")
        }
    return {
        "statusCode": 200,
        "body": json.dumps("Prefect flow triggered successfully!")
    }
```

