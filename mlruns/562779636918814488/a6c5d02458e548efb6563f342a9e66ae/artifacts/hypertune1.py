from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

data = load_breast_cancer()
x=pd.DataFrame(data.data, columns=data.feature_names)
y=pd.Series(data.target, name="target")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

rf=RandomForestClassifier(random_state=42)

paramgird={
    'n_estimators': [10,50,100],
    'max_depth': [None,10,20,30]
}

gridsearch=GridSearchCV(estimator=rf,param_grid=paramgird,cv=5,n_jobs=-1,verbose=2)

#run without mlflow
# gridsearch.fit(x_train, y_train)

# bestparams=gridsearch.best_params_
# bestscore=gridsearch.best_score_

# print(bestparams)
# print(bestscore)

mlflow.set_experiment("breast cancer")

with mlflow.start_run():
    gridsearch.fit(x_train, y_train)
    bestparams=gridsearch.best_params_
    bestscore=gridsearch.best_score_
    
    mlflow.log_params(bestparams)
    mlflow.log_metric("accuracy",bestscore)
    
    train_df=x_train.copy()
    train_df['target']=y_train
    
    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training")
    
    test_df=x_test.copy()
    test_df['target']=y_test
    
    test_df=mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"test")
    
    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(gridsearch.best_estimator_, "random_forest")
    
    mlflow.set_tag("author","ORBiT")
    
    print(bestparams)
    print(bestscore)