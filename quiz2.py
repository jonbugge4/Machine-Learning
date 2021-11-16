import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()


rating = pd.read_csv('gameratings.csv')


#print(rating.loc['title'])

data_train = rating.T.loc['console': 'violence']
data_train = data_train.T.values
#print(data_train)
#target_train = rating.T.loc['Target']
target_train = rating.Target.values
#print(target_train)

ersb = pd.read_csv('test_esrb.csv')


data_test = ersb.T.loc['console': 'violence']
data_test = data_test.T.values
#print(data_test)
#target_test = ersb.loc['Target']
target_test = ersb.Target.values
print(target_test)




from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

#THIS IS WHERE MACHINE LEARNING IS HAPPENING
knn.fit(X= data_train , y= target_train)  #Case sensititve

predicted = knn.predict(X= data_test) #dont need a target y because it predicts it
expected = target_test


#print(predicted[:20])
#print(expected[:20])


#Display Wrong (2)
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]

print(wrong)




''''''

