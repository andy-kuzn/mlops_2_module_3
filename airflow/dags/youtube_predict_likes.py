from airflow import DAG
from airflow.operators.bash import BashOperator
import datetime as dt
 
args = {
    "owner": "admin",
    "start_date": dt.datetime(2024, 2, 2),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=10),
    "depends_on_past": False,
}
 
with DAG(
    dag_id='youtube_predict_likes',
    default_args=args,
    tags=['youtube', 'score'],
    schedule="@hourly",
    catchup = True 
) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="python3 /home/andrey/project/scripts/get_data.py",
                            dag=dag)
    train_test_split = BashOperator(task_id='train_test_split',
                            bash_command="python3 /home/andrey/project/scripts/train_test_split.py",
                            dag=dag)  
    scale_data = BashOperator(task_id='scale_data',
                            bash_command="python3 /home/andrey/project/scripts/scale_data.py",
                            dag=dag)
    train_model = BashOperator(task_id='train_model',
                            bash_command="python3 /home/andrey/project/scripts/train_model.py",
                            dag=dag)
    test_model = BashOperator(task_id='test_model',
                            bash_command="python3 /home/andrey/project/scripts/test_model.py",
                            dag=dag)
    get_data >> train_test_split >> scale_data >> train_model >> test_model
