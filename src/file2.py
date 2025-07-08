import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import dagshub
dagshub.init(repo_owner='gbhatt7', repo_name='expwithmlflow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/gbhatt7/expwithmlflow.mlflow")

wine=load_wine()
X=wine.data
y=wine.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10,random_state=42)

max_depth=5
n_estimators=10

mlflow.set_experiment("EXPERIMENT 1")

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred=rf.predict(X_test)
    
    accuracy=accuracy_score(y_test,y_pred)
    
    cm=confusion_matrix(y_test, y_pred)
    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    plt.savefig("confusionmatrix.png")
    
    mlflow.log_artifact("confusionmatrix.png")
    mlflow.log_artifact(__file__)
    
    mlflow.set_tags({"Author": "ORBiT", "Project": "Wine Classification "})
    
    joblib.dump(rf, "rf_model.pkl")
    
    mlflow.log_artifact("rf_model.pkl")

    print(accuracy)