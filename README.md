### Lambda Function
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

'''

'''