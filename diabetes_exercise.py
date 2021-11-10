''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, model_selection

from sklearn.datasets import load_diabetes
diabetes = datasets.load_diabetes()

#how many sameples and How many features?
print(diabetes.data.shape)
#442 samples, 10 features


# What does feature s6 represent?
print(diabetes.DESCR)
# blood sugar level


#print out the coefficient
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11)

mymodel = LinearRegression()

mymodel.fit(X_train, Y_train)



#print out the intercept
print(mymodel.intercept_)

#test your model
predicted = mymodel.predict(X_test)
expected = Y_test

# create a scatterplot with regression line

plt.plot(expected, predicted, '.')


x = np.linspace(0,330, 100)
print(x)
y = x 
plt.plot(x,y)
plt.show()
