from surprise import Reader, Dataset, accuracy, SVD
from surprise.model_selection import train_test_split
from prefect import task
import numpy as np
import pandas as pd
import mlflow
import optuna


def objective(trial, trainset, testset):
    # 하이퍼파라미터의 범위 및 분포 지정
    n_factors = trial.suggest_int("n_factors", 50, 150)
    lr_all = trial.suggest_float("lr_all", 0.002, 0.01, log=True)
    reg_all = trial.suggest_float("reg_all", 0.02, 0.2, log=True)
    biased = trial.suggest_categorical("biased", [True, False])
    w1 = trial.suggest_float("w1", 0, 1)  #w1: RMSE 가중치, w2: MAE 가중치
    w2 = 1 - w1  

    # 모델 학습 및 검증
    model = SVD(n_factors=n_factors, lr_all=lr_all, reg_all=reg_all, biased=biased)
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    # 가중치를 적용한 평가지표 재계산. 이 값이 낮게 나오도록 학습 진행
    combined_score = w1 * rmse + w2 * mae

    return combined_score

@task
def train_svd(df_raw):
    mlflow.set_tracking_uri('http://localhost:5000')
    print(f'tracking URI: {mlflow.get_tracking_uri()}')

    mlflow.set_experiment('MLFLOW_Optuna_TEST_EXPERIMENT') 
    reader = Reader(rating_scale=(0,1))
    data = Dataset.load_from_df(df_raw, reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=12)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, trainset, testset), n_trials=100)  

    # Optuna로 찾은 best parma & best model로 재학습 및 평가진행
    best_params = study.best_params
    best_model = SVD(
        n_factors=best_params["n_factors"],
        lr_all=best_params["lr_all"],
        reg_all=best_params["reg_all"],
        biased=best_params["biased"]
    )
    
    best_model.fit(trainset)
    predictions = best_model.test(testset)
    
    with mlflow.start_run():
        for key, value in best_params.items():
            mlflow.log_param(key, value)
        
        mlflow.log_param("w1", study.best_params["w1"])
        mlflow.log_param("w2", 1 - study.best_params["w1"])
        
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(best_model, "best_svd_model")
    
    return 