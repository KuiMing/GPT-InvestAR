from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
with DAG(
    "prediction",
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="A test DAG",
    schedule="35 9 1 * *",
    start_date=datetime(2024, 5, 1),
    catchup=False,
    tags=["Monthly"],
) as dag:

    t6 = BashOperator(
        task_id="make-target",
        depends_on_past=False,
        bash_command="cd ../GPT-InvestAR && python3.9 make_targets_yf.py --config_path config.json --sqlite price.sqlite",
    )

    t7 = BashOperator(
        task_id="prediction",
        depends_on_past=False,
        bash_command="cd ../GPT-InvestAR && python3.9 predict_top_5.py",
    )
    t6 >> t7
