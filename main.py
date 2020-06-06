
#regresin logistica
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("winequality-red.csv", sep = ';')

#transformar la quality; 1 : buena calidad, 0: mala calidad 
count = 0
for row in dataset.quality:
    if( dataset.quality[count] > 6.5 ):
        dataset.quality[count] = 1
    else:
        dataset.quality[count] = 0
    count += 1

x = dataset.iloc[:, [2,3]].values # x:citric acid  y:residual sugar
y = dataset.iloc[:, 11].values    # quality


#dividir test y entrenamiento
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)


#se escalan los datos debido a que citric acid y residual sugar son muy disitintos
from sklearn.preprocessing import StandardScaler
sc_X    = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test  = sc_X.transform(x_test)

#ajustar modelo 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

yPred= classifier.predict(x_test)

#matrices de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,yPred) #recibe la 'y' real y la 'y' predecida

#
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,yPred)
print('accuracy : ', accuracy)

"""
#representación gráfica de los resultados del algoritmo en el conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Regresión Logistica')
plt.xlabel('Citric acid')
plt.ylabel('Residual sugar')
plt.legend()
plt.show()
"""
