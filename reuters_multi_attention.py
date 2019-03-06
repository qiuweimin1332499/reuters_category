from keras import backend as K
from keras.engine.topology import Layer

class Position_Embedding(Layer):
    def __init__(self, size=None, **kwargs):
        self.size = size  # 必须为偶数
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000.,2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1
        # K.arange不支持数据类型及长度，只好用这种方法生成，不过很奇怪为什么这样就能生成浮点类型的数据
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        return position_ij + x

    def compute_output_shape(self, input_shape):
            return input_shape

from keras.engine.topology import Layer

class Attention(Layer):

    def __init__(self,nb_head,size_per_head,**kwargs):
        self.nb_head=nb_head
        self.size_per_head=size_per_head
        self.output_dim=nb_head*size_per_head
        super(Attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.WQ=self.add_weight(name='WQ',
                                shape=(input_shape[0][-1], self.output_dim),
                                initializer='glorot_uniform',
                                trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention,self).build(input_shape)

    def call(self,x):
        Q_seq,K_seq,V_seq=x
        Q_seq=K.dot(Q_seq,self.WQ)
        Q_seq=K.reshape(Q_seq,(-1,K.shape(Q_seq)[1],self.nb_head,self.size_per_head))
        Q_seq=K.permute_dimensions(Q_seq,(0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        O_seq=K.batch_dot(Q_seq,K_seq,axes=[3,3])/self.size_per_head**0.5
        O_seq=K.softmax(O_seq)
        O_seq=K.batch_dot(O_seq,V_seq,axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq=K.reshape(O_seq,(-1,K.shape(O_seq)[1],self.output_dim))
        return O_seq

    def compute_output_shape(self, input_shape):
        return(input_shape[0][0],input_shape[0][1],self.output_dim)

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

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import *
from keras.models import Model
max_features=10000
maxlen=100
x_train=sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=sequence.pad_sequences(x_test,maxlen=maxlen)
#序列填充
S_inputs=Input(shape=(None,),dtype='int32')
embeddings=Embedding(max_features,128)(S_inputs)
#embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率
O_seq=Attention(4,8)([embeddings,embeddings,embeddings])#三输入多头注意力模型
O_seq=GlobalAveragePooling1D()(O_seq)
O_seq=Dropout(0.2)(O_seq)
outputs=Dense(46,activation='sigmoid')(O_seq)
model=Model(inputs=S_inputs,outputs=outputs)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

x_val=x_train[:1000]
partial_x_train=x_train[1000:]
y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=5,
                  batch_size=32,
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