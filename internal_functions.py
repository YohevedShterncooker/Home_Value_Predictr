import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main_calc():
   df = pd.read_excel("./HouseSold.xlsx", engine='openpyxl',
                   usecols=['date',
                            'age',
                            'MRT_station',
                            'conv_stores',
                            'latitude', 'longitude',
                            'price'])

   model = LinearRegression

   x = ['date','age','MRT_station','conv_stores','latitude', 'longitude']
   y = df['price']

   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

   #Train the model
   model.fit(x_train, y_train)

   y_pred = model.predict(x_test)
   print(y_pred)

   Error = mean_squared_error(y_test, y_pred)
   print("MSE:", Error)

   # show the point
   
   plt.plot(y_test, y_pred, '+b')
   # plt.plot(y_test, predicted_line, color='red')
   y_test_min = min(y_test)
   y_test_max = max(y_test)
   plt.plot(np.arange(y_test_min, y_test_max, 0.1), np.arange(y_test_min, y_test_max, 0.1), color='red')
   plt.xlabel("Prices")
   plt.ylabel("Predicted prices")
   plt.title("Actual Vs. Predicted prices")
   plt.tight_layout()
   plt.show()

      
   
   

  
  
