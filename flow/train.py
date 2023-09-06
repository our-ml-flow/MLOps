import os
import sys
repo_dir = os.path.abspath(__file__).split('/flow')[0]
print(repo_dir)
sys.path.append(f'{repo_dir}')

from prefect import flow, context
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from datetime import date, timedelta

from module.task import train_svd
from module.utils import get_raw_data


@flow
def svd_train_flow():
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=7)
    df_train_data = get_raw_data(start_date, end_date)
    train_svd(df_train_data)
    

if __name__=='__main__':
    
    deployment=Deployment.build_from_flow(
        flow=svd_train_flow,
        name='Train_svd_weekly',
        version=1.1,
        work_queue_name='training_agent',
        schedule=(CronSchedule(cron="15 11 * * MON", timezone="Asia/Seoul"))
    )
    deployment.apply()