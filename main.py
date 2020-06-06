
#regresin logistica
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("winequality-white.csv", sep = ';')

#nota: el dataset tiene escalada la calidad de 0 - 10; por lo cual transformo en memoria los datos de la col quality
#transformar la quality; 1 : buena calidad, 0: mala calidad 
count = 0
for row in dataset.quality:
    if( dataset.quality[count] >= 6.5 ):
        dataset.quality[count] = 1
        #print(dataset["sulphates"][count], dataset["alcohol"][count])
    else:
        dataset.quality[count] = 0
    count += 1

x = dataset.iloc[:, [9,10]].values # x:sulphates  y:alcohol
y = dataset.iloc[:, 11].values     # quality


#dividir test y entrenamiento: uso un 20% de test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


#se escalan los datos debido a que sulphates y alcohol son muy distintos
from sklearn.preprocessing import StandardScaler
scalar  = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test  = scalar.transform(x_test)

#ajustar modelo 
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

#acurracy
accuracy = clf.score(x_test, y_test)
print('accuracy : ', accuracy)

#datos de pureba: con x:sulphates  y:alcohol; ver las col del dataset winequality-white.csv y
#tomar valores de esa columna y cambiarlos en data para hacer pruebas
data = scalar.transform([[0.52, 12.4]]) #buena calidad: 0.52;12.4 mala calidad:0.63, 10.8
predict = clf.predict(data)
print(predict)

#representación gráfica de los resultados del algoritmo en el conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Regresión Logistica')
plt.xlabel('Sulphates')
plt.ylabel('Alcohol')
plt.legend()
plt.show()
