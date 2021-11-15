import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()


rating = pd.read_csv('gameratings.csv')
rating = rating.T

#print(rating.loc['title'])

data_train = rating.loc['title': 'violence']
#print(data_train)
target_train = rating.loc['Target']
#print(target_train)

ersb = pd.read_csv('test_esrb.csv')
ersb = ersb.T

data_test = ersb.loc['title': 'violence']
#print(data_test)
target_test = ersb.loc['Target']
#print(target_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

#THIS IS WHERE MACHINE LEARNING IS HAPPENING
knn.fit(X= data_train , y= target_train)  #Case sensititve

predicted = knn.predict(X= data_test) #dont need a target y because it predicts it
<<<<<<< HEAD
expected = target_test
=======
expected = target_test

print(predicted[:20])
print(expected[:20])

wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]

print(wrong)
''''''

>>>>>>> db57c389e5c12f511330dd9b8a4875d8517637af
