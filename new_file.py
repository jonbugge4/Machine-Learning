from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
print(iris.data.shape)
print(iris.target.shape)
print(iris.target_names)


data_train,data_test,target_train,target_test = train_test_split(
    iris.data, iris.target, random_state = 11)
print(data_train.shape)
print(target_train.shape)
print(data_test.shape)
print(target_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
#THIS IS WHERE MACHINE LEARNING IS HAPPENING
knn.fit(X= data_train , y= target_train)  #Case sensititve

predicted = knn.predict(X= data_test) #dont need a target y because it predicts it
expected = target_test

print(predicted[:20])
print(expected[:20])
print(iris.target_names)

predicted = [iris.target_names[x] for x in predicted]
expected = [iris.target_names[x] for x in expected]
print(predicted[:20])
print(expected[:20])

wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]

print(wrong)

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true = expected, y_pred = predicted)

print(confusion)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index = iris.target_names, columns = iris.target_names)

figure = plt2.figure(figsize=(7,6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
plt2.xlabel('Expected')
plt2.ylabel('Predicted')
plt2.show()