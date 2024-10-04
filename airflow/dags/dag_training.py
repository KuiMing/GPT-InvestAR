from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator

date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
with DAG(
    "training",
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
    schedule="0 12 1 1 *",
    start_date=datetime(2024, 3, 6),
    catchup=False,
    tags=["Yearly"],
) as dag:

    # t1, t2 and t3 are examples of tasks created by instantiating operators

    t6 = BashOperator(
        task_id="make-target",
        depends_on_past=False,
        bash_command="cd ../GPT-InvestAR && python3.9 make_targets_yf.py --config_path config.json --sqlite price.sqlite",
    )

    t8 = BashOperator(
        task_id="training",
        depends_on_past=False,
        bash_command="cd ../GPT-InvestAR && python3.9 update_model.py",
    )

    t6 >> t8
