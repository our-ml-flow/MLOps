from prefect_sqlalchemy import SqlAlchemyConnector
from prefect.blocks.system import JSON
from sqlalchemy import text

from prefect import task
import requests
import pandas as pd


@task
def get_raw_data(start_date, end_date):
    
    engine=SqlAlchemyConnector.load("gcp-mlops-sql-postgres").get_engine()
    connection = engine.connect()

    try:
        query = f""" WITH owner_token_info AS (
                        SELECT 
                            owner_address,
                            collection_name AS collection,
                            num_distinct_tokens_owned AS num_token,
                            sum(num_distinct_tokens_owned) OVER (PARTITION BY owner_address) AS tot,
                            data_created_at::date AS created_at
                        FROM alchemy_collection_for_buyer
                        WHERE data_created_at::date BETWEEN {start_date} AND {end_date}'
                        )
                        SELECT 
                            owner_address, 
                            collection,
                            sum(num_token/tot) AS token_ratio
                        FROM owner_token_info
                        GROUP BY owner_address, collection, created_at
                        ORDER BY owner_address;"""
        
        result = connection.execute(text(query))
        rows=result.fetchall()
        df=pd.DataFrame(rows)
    except Exception as e:
        print("error", e)
    return df