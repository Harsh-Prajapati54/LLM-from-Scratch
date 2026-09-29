import mlflow

mlflow.set_experiment("LLM_From_Scratch")

with mlflow.start_run():

    mlflow.log_param("test", "MLflow working")

    mlflow.log_metric("test_metric", 1.0)

print("MLflow test completed!")