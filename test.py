#regresin logistica
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("winequality-red.csv", sep = ';')

"""
print( dataset.quality[8] )
print( np.equal(dataset.quality[8], 7 ))
"""


count = 0
for row in dataset.quality:
    if( dataset.quality[count] > 6.5 ):
        dataset.quality[count] = 1
    else:
        dataset.quality[count] = 0
    print(dataset.quality[count])
    count += 1


#dataset.loc[dataset.quality >  6.5, 'quality'] = 1
#dataset.loc[dataset.quality <= 6.5, 'quality', 'quality'] = 0

#y = dataset.iloc[:, 11].values

#print(dataset.quality)
#dataset.loc[ dataset["quality"] > 6.5, "quality"] = 1
#dataset.loc[ dataset["quality"] <= 6.5, "quality"] = 0

#dataset.loc[(dataset["quality"] > 6.5),'quality'] = 1
#dataset.loc[(dataset["quality"] <= 6.5),'quality'] = 0

#print(dataset["quality"])

#x = dataset.iloc[:, :-1 ]
#y = dataset.iloc[:, 11].values

#good_wine = dataset[ y > 6.5]
#bad_wine  = dataset[ y <= 6.5]

#arr = np.concatenate((good_wine, bad_wine))
#print(arr[:,11])

#plt.scatter(good_wine.iloc[:,0], good_wine.iloc[:,1],s=10, label='Good Wine')
#plt.scatter(bad_wine.iloc[:,0], bad_wine.iloc[:,1],s=10, label='Bad Wine')
#plt.xlabel('Fixed Acidity')
#plt.ylabel('Volatile Acidity')
#plt.legend()
#plt.show()
#print('good wines:',good_wine.shape)
#print('bad wines:',bad_wine.shape)
