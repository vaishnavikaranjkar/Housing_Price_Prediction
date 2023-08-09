import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("test1")

house_data = pd.read_csv("datasets/train.csv")
# print(df.columns)
area=house_data['LotArea']
price=house_data['SalePrice']

# because ML handles arrays and not data frames
x = np.array(area).reshape(-1,1)
y = np.array(price).reshape(-1,1)

# using LR and fit() in training
model=LinearRegression()
model.fit(x,y)

# MSE and R
mse = mean_squared_error(x,y)
print("MSE: ", math.sqrt(mse))
print("R squared: ", model.score(x,y))

# slope
print("Slope: ", model.coef_[0])
# intercept
print("Intercept: ", model.intercept_[0])

# plot
plt.scatter(x,y,color='green')
plt.plot(x,model.predict(x), color='black')
plt.show()

# predict value of house with area - 2000
print("Prediction: ", model.predict([[2000]]))
