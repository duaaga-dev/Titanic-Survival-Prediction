import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("tested.csv")
data.info()
print(data.isnull().sum())


#Data Cleaning and feature engineering
def preprocess_data(df):
   df.drop(columns=["PassengerId","Name","Ticket","Cabin"], inplace=True)


   df ["Embarked"].fillna("S")
   df.drop(columns=["Embarked"], inplace=True)


 

#Convert Gender
   df["Sex"] = df["Sex"].map({'male' :1 , "female": 0})


 #Feature Engineering 
   df["FamilySize"] = df["SibSp"] + df["Parch"]
   df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
   df["Fare"].fillna(df["Fare"].median())

   df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
   df["AgeBin"] = pd.cut(df["Age"] , bins=[0,12,20,40,60, np.inf], labels=False)

   return df

#Fill in Missing Ages
def fill_missing_ages(df):
   age_fill_map = {}
   for Pclass in df["Pclass"].unique():
    if Pclass not in age_fill_map:
        age_fill_map[Pclass] = df[df["Pclass"] == Pclass]["Age"].median() 

   df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]]
                         if pd.isnull(row["Age"]) 
                         else row["Age"] ,
                         axis= 1)
   return df


data = fill_missing_ages(data)
data = preprocess_data(data)

print(data.isnull().sum())


#Create Features / Target Variables (Make Flashcards)
data = data.fillna(0)
x = data.drop(columns=["Survived"])
y = data["Survived"]


X_train, X_test, Y_train, Y_test = train_test_split(
x, y, test_size=0.2, random_state=42
)

#ML Preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


#Hayperparemeter Tuning - KNN
def tune_model(X_train ,Y_train):
   param_grid = {
      "n_neighbors":range(1,21),
       "metric": ["euclidean" , "manhattan", "minkowski"],
         "weights" : ["uniform", "distance"] 
           }


   model = KNeighborsClassifier()
   grid_search = GridSearchCV(model, param_grid, cv=5 , n_jobs=-1)
   grid_search.fit(X_train, Y_train)
   return grid_search.best_estimator_


best_model = tune_model(X_train, Y_train)

#Predictions and Evaulate
def evaluate_model(model, X_test, Y_test):
   prediction = model.predict(X_test)
   accuracy = accuracy_score(Y_test, prediction)
   matrix = confusion_matrix(Y_test, prediction)
   return accuracy, matrix


accuracy, matrix = evaluate_model(best_model, X_test, Y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'confusion Matrix:')
print(matrix)



#Plot
def plot_model(matrix):
   plt.figure(figsize=(10,7))
   sns.heatmap(matrix, annot=True, fmt="d" , xticklabels=["Survived", "Not Survives"], yticklabels=["Not Survived", "Survived"])
   plt.title("Confusion Matrix")
   plt.xlabel("Predicted Value")
   plt.ylabel("True Values")
   plt.show()


plt.savefig("confusion_matrix.png")
plot_model(matrix)