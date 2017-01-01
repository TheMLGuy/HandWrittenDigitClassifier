#
#@author: Ashwin
#"""



from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing

seed=7
np.random.seed(seed)
digits=pd.read_csv('train.csv')
testSet=pd.read_csv('test.csv')
digits=np.array(digits)
testSet=np.array(testSet)
label=digits[:,0]
#define training set and testset
trainSet=digits[:,1:(digits.shape[1]+1)]

#standardizing feauture set
std_scale=preprocessing.StandardScaler().fit(trainSet)
trainSet_std=std_scale.transform(trainSet)
testSet_std=std_scale.transform(testSet)
#one hot encoding scheme for output label
numclasses=label.shape[1]
numlabels=label.shape[0]
indexOffset=np.arange(numlabels)*numclasses
one_hot_vector=np.zeros((numlabels,numclasses))
one_hot_vector.flat[indexOffset+label.ravel()]=1
label_one_hot=one_hot_vector
label_one_hot.astype(np.uint8)

#define model and stack layers on top
model=Sequential()
model.add(Dense(390,input_dim=784,init='uniform',activation='relu'))
model.add(Dense(10,init='uniform'))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])

model.fit(trainSet_std,label_one_hot,nb_epoch=5,batch_size=10)

eval=model.evaluate(trainSet_std,label_one_hot)
print("%s: %.2f%% " % (model.metrics_names[1], eval[1]*100))

#prediction chunk
prediction=model.predict_classes(testSet_std)
print(prediction)
print("------------------------------------------------------")