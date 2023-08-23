import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#import your dataset
disease = datasets.load_diabetes()
#print dataset
print(disease)

#take the data
#newaxis to enable us to plot by segregating the data
disease_x = disease.data[:, np.newaxis, 2]

#split the data
disease_x_train = disease_x[:-30]
disease_x_test = disease_x[-20:]

disease_y_train = disease.target[:-30]
disease_y_test = disease.target[-20:]

#generate your model
reg = linear_model.LinearRegression()

#fit the data into the model
reg.fit(disease_x_train, disease_y_train)

#prediction
y_predict = reg.predict(disease_x_test)

#calculate accuracy
accuracy = mean_squared_error(disease_y_test,y_predict)

#print accuracy
print("Accuracy: ", accuracy)

#print the weights(coefficients) and coefficient
weights = reg.coef_
intercept = reg.intercept_
print("Coefficients: ", weights, "Intercept: ", intercept)

#plotting
plt.scatter(disease_x_test, disease_y_test)
plt.plot(disease_x_test, y_predict)
plt.show()
