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
        query = f"""
                WITH 
                    top_10_addresses AS (
                        SELECT TRIM(nft_contract_address) AS top10
                        FROM dune_nft_trades
                        WHERE block_time BETWEEN '{start_date}' AND '{end_date}' 
                        GROUP BY nft_contract_address
                        ORDER BY sum(amount_usd) DESC 
                        LIMIT 10
                    ),
                    owner_token_info AS (
                        SELECT 
                            owner_address,
                            TRIM(contract->>'address') AS address,
                            collection_name AS collection,
                            num_distinct_tokens_owned AS num_token,
                            CASE 
                                WHEN contract->>'address' IN (SELECT * FROM top_10_addresses) THEN 1.1 
                                ELSE 1.0 
                            END AS weight
                        FROM alchemy_collection_for_buyer
                        WHERE data_created_at::date BETWEEN '{start_date}' AND '{end_date}'
                    ),
                    rawdata AS(
                    SELECT owner_address,
                    collection,
                    num_token,
                    sum(num_token) OVER (PARTITION BY owner_address) AS tot,
                    weight
                    FROM owner_token_info
                    )
                    SELECT owner_address,collection,
                    (num_token/tot)*weight AS ratio
                    FROM rawdata;
                """
        
        result = connection.execute(text(query))
        rows=result.fetchall()
        print(rows)
        df=pd.DataFrame(rows)
    except Exception as e:
        print("error", e)
    finally:
        connection.close()
    return df
