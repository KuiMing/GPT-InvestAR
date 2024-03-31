from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator

date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
with DAG(
    "get-annual-report-pdf",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
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
) as dag:

    # t1, t2 and t3 are examples of tasks created by instantiating operators

    t6 = BashOperator(
        task_id="make-target",
        depends_on_past=False,
        bash_command="python3.9 make_targets_yf.py --sqlite price.sqlite",
    )

    t7 = BashOperator(
        task_id="prediction",
        depends_on_past=False,
        bash_command="python3.9 predict_top_5.py ",
    )
    t6 >> t7
