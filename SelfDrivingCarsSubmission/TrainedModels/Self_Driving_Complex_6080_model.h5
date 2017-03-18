
# coding: utf-8

# In[6]:


import numpy as np
import scipy.misc
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# In[7]:

np.random.seed(123) 


# In[8]:

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.txt
#read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.08)]
train_ys = ys[:int(len(xs) * 0.08)]

val_xs = xs[-int(len(xs) * 0.02):]
val_ys = ys[-int(len(xs) * 0.02):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out


# In[ ]:




# In[9]:

model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(66, 200,3), border_mode='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(48, 5, 5, border_mode='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


# In[10]:

model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print(model.summary())


# In[ ]:

batch_size=32
for epoch in range(2):
    print 'epoch %d' % (epoch)
    for i in range(len(xs)//batch_size):
        x_batch,y_batch=LoadTrainBatch(batch_size)
        X_train=np.array(x_batch)
        Y_train=np.array(y_batch)
        model.train_on_batch(X_train,Y_train,class_weight=None, sample_weight=None)


