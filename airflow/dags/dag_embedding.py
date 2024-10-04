from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
with DAG(
    "get-annual-report-pdf",
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="A test DAG",
    schedule="5 9 1 * *",
    start_date=datetime(2024, 3, 6),
    catchup=False,
    tags=["Daily"],
    max_active_runs=1,
) as dag:

    t3 = BashOperator(
        task_id="pdf-to-embedding",
        depends_on_past=False,
        bash_command="cd ../GPT-InvestAR && python3.9 embeddings_inference.py --config_path config.json --save_directory inference_chroma/",
    )

    t5 = BashOperator(
        task_id="get-feature",
        depends_on_past=False,
        bash_command="cd ../GPT-InvestAR && python3.9 gpt_score_inference.py --save_directory inference_chroma/",
    )

    t3 >> t5
