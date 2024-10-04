from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
with DAG(
    "AR-price",
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="A test DAG",
    schedule="5 6 * * 1-5",
    start_date=datetime(2024, 4, 2),
    catchup=False,
    tags=["Daily"],
) as dag:

    t1 = BashOperator(
        task_id="get-html",
        bash_command="cd ../edgar-crawler && python3.9 edgar_crawler.py && cd datasets/INDICES && rm `ls | grep tsv`",
    )

    t2 = BashOperator(
        task_id="html-to-pdf",
        depends_on_past=False,
        bash_command="cd ../GPT-InvestAR && python3.9 convert_html_to_pdf.py --config_path config.json",
        retries=3,
    )

    t4 = BashOperator(
        task_id="get-price",
        depends_on_past=False,
        bash_command="cd ../GPT-InvestAR && python3.9 get_price.py --config_path config.json",
    )

    t1 >> t2 >> t4
