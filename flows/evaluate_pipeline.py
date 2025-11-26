from prefect import flow, task, get_run_logger
import mlflow
from mlflow import MlflowClient
from prefect.blocks.system import Secret

@task
def get_mlflow_client():
    tracking_uri = Secret.load("mlflow-server").get()
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()

@task
def get_latest_metrics_for_run(client: MlflowClient, run_id: str) -> dict:
    run = client.get_run(run_id)
    return dict(run.data.metrics)

@task
def get_production_run_info(client: MlflowClient, model_name: str):
    latest = client.get_latest_versions(model_name, stages=["Production"])
    if not latest:
        raise ValueError(f"No version found for model '{model_name}' in stage 'Production'")
    prod_mv = latest[0]
    return prod_mv.run_id, prod_mv.version

@task
def print_and_compare_metrics(
    incoming_metrics: dict,
    prod_metrics: dict,
) -> bool:
    """
    Compare metrics and return True if incoming model should be promoted to production.
    
    Improvement margins: mAP50-95 (+2%), mAP50 (+2%), Precision (+1%), Recall (+1%)
    """
    logger = get_run_logger()
    logger.info("\nComparison (incoming vs production):")
    all_keys = set(incoming_metrics.keys()) | set(prod_metrics.keys())
    for k in sorted(all_keys):
        inc = incoming_metrics.get(k)
        prod = prod_metrics.get(k)
        logger.info(f"  {k}: incoming={inc} | production={prod}")

    # Extract key metrics
    inc_map50_95 = incoming_metrics.get('metrics/mAP50-95B', 0.0)
    prod_map50_95 = prod_metrics.get('metrics/mAP50-95B', 0.0)
    inc_map50 = incoming_metrics.get('metrics/mAP50B', 0.0)
    prod_map50 = prod_metrics.get('metrics/mAP50B', 0.0)
    inc_precision = incoming_metrics.get('metrics/precisionB', 0.0)
    prod_precision = prod_metrics.get('metrics/precisionB', 0.0)
    inc_recall = incoming_metrics.get('metrics/recallB', 0.0)
    prod_recall = prod_metrics.get('metrics/recallB', 0.0)
    
    # Calculate improvements
    map50_95_diff = inc_map50_95 - prod_map50_95
    map50_diff = inc_map50 - prod_map50
    precision_diff = inc_precision - prod_precision
    recall_diff = inc_recall - prod_recall
    
    # Check thresholds
    map50_95_passed = map50_95_diff >= 0.02
    map50_passed = map50_diff >= 0.02
    precision_passed = precision_diff >= 0.01
    recall_passed = recall_diff >= 0.01
    
    logger.info(f"\n⭐️ Model Comparison (Production vs Incoming)")
    logger.info(f"mAP50-95:  {prod_map50_95:.4f} → {inc_map50_95:.4f} ({map50_95_diff:+.4f}) {'✅' if map50_95_passed else '❌'}")
    logger.info(f"mAP50:     {prod_map50:.4f} → {inc_map50:.4f} ({map50_diff:+.4f}) {'✅' if map50_passed else '❌'}")
    logger.info(f"Precision: {prod_precision:.4f} → {inc_precision:.4f} ({precision_diff:+.4f}) {'✅' if precision_passed else '❌'}")
    logger.info(f"Recall:    {prod_recall:.4f} → {inc_recall:.4f} ({recall_diff:+.4f}) {'✅' if recall_passed else '❌'}")
    
    # Decision: mAP50-95 must improve by 2% AND at least one other metric improves
    other_metric_passed = map50_passed or precision_passed or recall_passed
    should_promote = map50_95_passed and other_metric_passed
    
    logger.info(f"\n⭐️ Threshold: mAP50-95 ≥+2% AND (mAP50 ≥+2% OR Precision ≥+1% OR Recall ≥+1%)")
    logger.info(f"   mAP50-95 requirement: {'✅ PASS' if map50_95_passed else '❌ FAIL'}")
    logger.info(f"   Other metric requirement: {'✅ PASS' if other_metric_passed else '❌ FAIL'}")
    logger.info(f"\n{'✅ PROMOTE TO PRODUCTION' if should_promote else '❌ DO NOT PROMOTE'}")
    
    return should_promote

@task
def promote_model_to_production(
    client: MlflowClient,
    incoming_run_id: str,
    model_name: str,
    current_prod_version: int
):
    """Promote the incoming model to production and archive the current production model."""
    logger = get_run_logger()
    all_versions = client.search_model_versions(f"name='{model_name}'")
    incoming_version = None
    
    for mv in all_versions:
        if mv.run_id == incoming_run_id:
            incoming_version = mv.version
            break
    
    if incoming_version is None:
        raise ValueError(f"Could not find model version for run_id {incoming_run_id}")
    
    # Archive current production and promote new model
    client.transition_model_version_stage(
        name=model_name,
        version=current_prod_version,
        stage="Archived",
        archive_existing_versions=False
    )
    
    client.transition_model_version_stage(
        name=model_name,
        version=incoming_version,
        stage="Production",
        archive_existing_versions=False
    )
    
    logger.info(f"✅ Promoted version {incoming_version} to Production (archived v{current_prod_version})")
    return incoming_version

@flow(name="evaluate_pipeline", log_prints=True)
def evaluate_pipeline(
    incoming_run_id: str,
    model_name: str,
):
    client = get_mlflow_client()
    logger = get_run_logger()

    incoming_metrics = get_latest_metrics_for_run(client, incoming_run_id)
    prod_run_id, prod_version = get_production_run_info(client, model_name)
    prod_metrics = get_latest_metrics_for_run(client, prod_run_id)


    logger.info(f"\n### Incoming run_id: {incoming_run_id}")
    logger.info(f"\n### Production model version: {prod_version}, run_id: {prod_run_id}")


    should_promote = print_and_compare_metrics(
        incoming_metrics, 
        prod_metrics
    )
    
    if should_promote:
        promote_model_to_production(client, incoming_run_id, model_name, prod_version)
    else:
        logger.info(f"Keeping version {prod_version} in production.")


if __name__ == "__main__":
    evaluate_pipeline(
        incoming_run_id="8360a84aeb0f40a0943064e117ad31d0",
        model_name="fire_vs_smoke_yolov8",
    )