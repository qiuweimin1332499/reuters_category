
from keras.datasets import reuters
(train_data, train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

import numpy as np
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

def to_one_hot(labels,dimension=46):
    results=np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label]=1
    return results
one_hot_train_labels=to_one_hot(train_labels)
one_hot_test_labels=to_one_hot(test_labels)

from keras import models
from keras import layers
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(128,activation='relu',))
model.add(layers.Dense(46,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

x_val=x_train[:1000]
partial_x_train=x_train[1000:]
y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=9,
                  batch_size=512,
                  validation_data=(x_val, y_val))
#results=model.evaluate(x_test, one_hot_test_labels)
#print(results)

#import matplotlib.pyplot as plt
#loss=history.history['loss']
#val_loss=history.history['val_loss']
#epochs=range(1,len(loss)+1)
#plt.plot(epochs,loss,'bo',label='training loss')
#plt.plot(epochs,val_loss,'b',label='validation loss')
#plt.title('training and validation loss')
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.legend()
#plt.show()

import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='training acc')
plt.plot(epochs,val_acc,'b',label='validation acc')
plt.title('training and validation acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()