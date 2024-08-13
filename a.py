import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def training_model():
    df = pd.read_csv("./AGG-table.csv", usecols=['avg_jitter', 'avg_package_lost', 'avg_rtt', 'SCORE1'])
    
    df = df.dropna()

    #Define features and target
    features = ['avg_jitter', 'avg_rtt', 'avg_package_lost']
    target = 'SCORE1'

    #x = df[['avg_jitter', 'avg_package_lost', 'avg_rtt']]
    #y = df['SCORE1']

    x = df[features]
    y = df[target]

    # Scale the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x_scaled, y)

    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.15, random_state=42)

    model = LogisticRegression()
    
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(y_pred)

    Error = mean_squared_error(y_test, y_pred)
    print("MSE:", Error)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    pd.set_option('display.max_rows', 100)  
    print(df)

    Error = mean_squared_error(y_test, y_pred)
    print("MSE:", Error)

    data = pd.DataFrame({
    'avg_jitter': [0.02],
    'avg_package_lost': [0.01],
    'avg_rtt': [0.02],
  
    })
    print(model.predict(data))

    w = model.coef_
    print(w)
    
    

training_model()



        
   
    