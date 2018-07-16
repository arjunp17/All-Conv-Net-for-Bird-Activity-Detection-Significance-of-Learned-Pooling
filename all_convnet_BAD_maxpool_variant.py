from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, MaxPooling1D, Conv1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import keras

initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2)


model = Sequential()

#conv1
model.add(ZeroPadding2D((2,2),input_shape=(40,500,1)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#maxpool1
model.add(MaxPooling2D((5,1), strides=(5,1)))
model.add(Dropout(0.50))
#conv2

model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#maxpool2
model.add(MaxPooling2D((2,1), strides=(2,1)))
model.add(Dropout(0.50))

#conv3
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#maxpool3
model.add(MaxPooling2D((2,1), strides=(2,1)))
model.add(Dropout(0.50))

#conv4
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#maxpool4
model.add(MaxPooling2D((2,1), strides=(2,1)))
model.add(Dropout(0.50))

#stacking(reshaping)
model.add(Reshape((500, 16)))

#temporal squeeing
model.add(MaxPooling1D((500), strides=(1)))
model.add(Dropout(0.50))

#fully connected layers
model.add(Flatten())
model.add(Dense(196, activation='sigmoid'))
model.add(Dropout(0.50))
model.add(Dense(2, activation='softmax', name='predictions'))
model.summary()

#train_data
classes = 2
feature = np.load('../feature_train.npy')
label = np.load('../label_train.npy')
label = to_categorical(label, 2)
opt = Adam(decay = 1e-6)
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.00, shuffle=True)

# compile model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#fit the model
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,verbose=2)

#save_model
model.save('all_convnet_BAD_maxpool_variant.h5') 

#test_data
test_class_file = np.loadtxt('../test_class_file.txt',dtype='str')
test_data = np.load('../feature_test.npy')

#test_label_predict
clas_labels = model.predict_classes(test_data, batch_size=1, verbose=0)
pred_probs = model.predict_proba(test_data, batch_size=1, verbose=0)

#save_test_labels_&_probs
test_class_file=np.array(test_class_file)
pred_probs=np.array(pred_probs)
np.savetxt('allconvnet_BAD_maxpool_variant',np.c_[class_file,pred_probs[:,0],pred_probs[:,1]],fmt='%s')
