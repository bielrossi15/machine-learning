from sklearn import tree

verde = 0
vermelha = 1
uva = 0
ameixa = 1

pomar = [[10, verde], [12, verde], [13, verde], [20, vermelha], [25, vermelha], [27, vermelha]]

resultado = [uva, uva, uva, ameixa, ameixa, ameixa]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(pomar, resultado)

peso = input('peso: ')
cor = input('cor: ')

resultadoUser = clf.predict([[peso, cor]])

if resultadoUser == 0:
    print('Uva')

else :
    print('Ameixa')