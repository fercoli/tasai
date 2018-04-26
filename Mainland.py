import numpy as np
import pandas as pd
from Pricer import Builder
#Let's check this out
##pandas data processing
UF = 27000
data = pd.read_csv("databasefinal.csv")


for i in range(len(data)):
    if data.iloc[i,4] == "CLP":
        data.iat[i,5] = data.iloc[i,5]/UF
i = None

y = data.iloc[:,5]
newcols = list(data.columns)
newcols.pop(4)
newcols.pop(4)


data = data.reindex(columns = newcols)

soso = data.values

#encodeando
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_enconder_1 =  LabelEncoder()
soso[:,1] = label_enconder_1.fit_transform(soso[:,1])
label_enconder_4 =  LabelEncoder()
soso[:,4] = label_enconder_4.fit_transform(soso[:,4])
onehotencoder = OneHotEncoder(categorical_features = [1])
soso = onehotencoder.fit_transform(soso).toarray()

soso = soso[:,1:]

xy = y.values
xy = xy.astype(np.float64).reshape(-1,1)

y = y.values.reshape(1,-1)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

Thescaly = StandardScaler()
fff = MinMaxScaler(feature_range=(0,1))
otrososo = Thescaly.fit_transform(soso)
xy = fff.fit_transform(xy)
    

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(otrososo, xy, test_size = 0.2, random_state = 0)


Brainy = Builder(3, [10,20,1], ["relu","relu","sigmoid"],17)

Brainy.fit(X_train, y_train, batch_size = 5, epochs = 1000)


##we use piton
#a = open("database.csv")
#dataski = []
#useless = 0
#for i in a:
#    if useless == 0:
#        useless += 1
#        continue
#    temp = i.split(",")[:13]
#    temp[0] = int(temp[0])
#    temp[2] = int(temp[2])
#    temp[3] = int(temp[3])
#    temp[5] = int(temp[5])
#    temp[7] = int(temp[7])
#    temp[8] = int(temp[8])
#    temp[9] = float(temp[9])
#    temp[10] = float(temp[10])
#    dataski.append(temp)
#
#truedata = []
#for i in dataski:
#    if i[4] == "CLP":
#        i.pop(4)
#    else:
#        i[5] = i[5]*UF
#        i.pop(4)
#    truedata.append(i)
#    print(i)
#
