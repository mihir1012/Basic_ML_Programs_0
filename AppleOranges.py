from sklearn import tree
#Training Data
features = [[140,1],[130,1],[150,0],[170,1]]#0 for bumpy, 1 for smooth
labels = [0,0,1,1] #0 for apple , 1 for orange

clf =  tree.DecisionTreeClassifier()

clf = clf.fit(features,labels)

print(clf.predict([[150,0]]))
