from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator

date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
with DAG(
    "feature-price",
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
    schedule="5 6 * * 1-5",
    start_date=datetime(2024, 3, 6),
    catchup=False,
    tags=["Daily"],
) as dag:

    # t1, t2 and t3 are examples of tasks created by instantiating operators
    t1 = BashOperator(
        task_id="get-html",
        bash_command="python3.9 edgar_crawler.py && rm `ls datasets/INDICES`",
    )

    t2 = BashOperator(
        task_id="html-to-pdf",
        depends_on_past=False,
        bash_command="python3.9 convert_html_to_pdf.py --config_path config.json",
        retries=3,
    )

    t3 = BashOperator(
        task_id="pdf-to-embedding",
        depends_on_past=False,
        bash_command=f"python3.9 embeddings_inference.py --config_path config.json --save_directory /home/ben/sdb1/GPT-InvestAR/inference_chroma/",
    )

    t4 = BashOperator(
        task_id="get-price",
        depends_on_past=False,
        bash_command=f"python3.9 get_price.py --config_path config.json --start {datetime.now().strftime('%Y-%m-%d')}",
    )

    t5 = BashOperator(
        task_id="get-feature",
        depends_on_past=False,
        bash_command="python3.9 gpt_score_inference.py --save_directory /home/ben/sdb1/GPT-InvestAR/inference_chroma/",
    )

    t1 >> t2 >> t3 >> t5
    t2 >> t4
