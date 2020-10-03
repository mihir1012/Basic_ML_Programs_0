from sklearn import tree
#Training Data, where keeping labels as String

features = [[300,2],[450,2],[200,8],[150,9]]#horsepower and number of seats

labels =["sports-car","sports-car","minivan","minivan"]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features, labels)

print(clf.predict([[400,1]]))
