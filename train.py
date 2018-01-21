from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.backend import image_data_format
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras import regularizers
from keras.layers import MaxPooling1D
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Conv1D
from keras.optimizers import Adadelta
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dropout
import keras
import pickle
import sys
from scipy.signal import freqz

########################################################

print('Loading file...', end='')
with open('comp_data.pkl','rb') as f:
	((x_train,y_train),(x_test,y_test)) = pickle.load(f)
print('Done.')

########################################################

def categorize(target_in):
    target_out = [int(i) for i in target_in]
    #for i in target_in:
    #    dist = [np.abs(i-j) for j in range(10)]
    #	target_out.append(np.argmax(dist))
    return target_out

#######################################################

def test_eval(model,x_test,y_test,test_size=0.1):
    test_size = int(test_size*len(x_test))
    return np.mean([model.evaluate(np.array([x_test[j]]), np.array([y_te[j]]),verbose=0)[0] for j in np.random.randint(0,len(x_test),size=(test_size,))])

#######################################################

y_tr = keras.utils.to_categorical(categorize(y_train),10)
y_te = keras.utils.to_categorical(categorize(y_test),10)
batch_size=1
num_classes=10
epochs=2

########################################################

#Defining layers
input_shape = (None,3)
first_layer = Conv1D(filters=96,kernel_size=11,strides=2,padding='causal', \
                 activation='relu',use_bias=True,kernel_initializer='TruncatedNormal', \
                 bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01), \
                 bias_regularizer=regularizers.l2(0.01),input_shape=input_shape)
layer2 = Conv1D(filters=256,kernel_size=5,strides=2,padding='causal', \
                 activation='relu',use_bias=True,kernel_initializer='TruncatedNormal', \
                 bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01), \
                 bias_regularizer=regularizers.l2(0.01))
layer3 = Conv1D(filters=384,kernel_size=3,strides=2,padding='causal', \
                 activation='relu',use_bias=True,kernel_initializer='TruncatedNormal', \
                 bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01), \
                 bias_regularizer=regularizers.l2(0.01))
layer4 = Conv1D(filters=384,kernel_size=3,strides=2,padding='causal', \
                 activation='relu',use_bias=True,kernel_initializer='TruncatedNormal', \
                 bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01), \
                 bias_regularizer=regularizers.l2(0.01))
layer5 = Conv1D(filters=384,kernel_size=3,strides=2,padding='causal', \
                 activation='relu',use_bias=True,kernel_initializer='TruncatedNormal', \
                 bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01), \
                 bias_regularizer=regularizers.l2(0.01))
layer6 = Dense(100, activation='relu', use_bias=True, kernel_initializer='TruncatedNormal', \
                bias_initializer='zeros')
layer7 = Dense(100, activation='tanh', use_bias=True, kernel_initializer='TruncatedNormal', \
                bias_initializer='zeros')
layer8 = Dense(100, activation='tanh', use_bias=True, kernel_initializer='TruncatedNormal', \
                bias_initializer='zeros')
layer9 = Dense(10, activation='softmax', use_bias=True, kernel_initializer='TruncatedNormal', \
                bias_initializer='zeros')

#################################################################

#Model declaration and compilation
print('Building model...',end='')
model=Sequential()
model.add(first_layer)
model.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
model.add(layer2)
model.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
model.add(layer3)
model.add(layer4)
model.add(layer5)
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.25))
model.add(layer6)
model.add(layer7)
model.add(layer8)
model.add(layer9)
model.compile(loss=categorical_crossentropy, \
              optimizer=Adadelta(), \
              metrics=['accuracy'])
print('Done.')

###################################################################

print('Starting training:')
tr_loss=[]
te_loss=[]
eval_time=10
for ep in range(30):
	print('Running epoch %d:'%ep)
	for i in range(len(x_train)):
	    tr_loss.append(model.fit(np.array([x_train[i]]), np.array([y_tr[i]]),
	          batch_size=batch_size,
        	  epochs=epochs,
	          verbose=0,
	             shuffle=False).history['loss'][1])
	    if i%eval_time == 0:
	        te_loss.append(test_eval(model,x_test,y_test))
	    else:
	        te_loss.append(te_loss[-1])
	    sys.stdout.write('Percentage: %d %% (%d/%d)\r'%(int(100*(i+1)/len(x_train)),i+1,len(x_train)))
	    sys.stdout.flush()
	print('\n',end='')
	print('\nRunning final test evaluation...',end='')
	te_loss[-1]=np.mean([model.evaluate(np.array([x_test[j]]), np.array([y_te[j]]),verbose=0)[0] for j in range(len(x_test))])
print('Done.')

########################################################

print('Genrating loss curve...',end='')
plt.figure(figsize=(10,10))
plt.plot(tr_loss)
plt.plot(te_loss)
plt.xlabel('Training iteration')
plt.ylabel('Loss value.')
plt.legend(['Training Sample loss','Mean test validation loss'])
plt.savefig('loss_curve.png')
print('Done.')
print('Saving training and test losses...',end='')
with open('tr_loss.pkl','wb') as f:
	pickle.dump(tr_loss,f,protocol=pickle.HIGHEST_PROTOCOL)
with open('te_loss.pkl','wb') as f:
	pickle.dump(te_loss,f,protocol=pickle.HIGHEST_PROTOCOL)
print('Done.')

########################################################


########################################################

filters = first_layer.get_weights()
print('Genrating filter map...',end='')
plt.figure(figsize=(20,20))
for i in range(9):
    for j in range(9):
        fno=i*9 + j
        tempf1 = []
        tempf1.append(filters[1][fno])
        tempf2 = []
        tempf2.append(filters[1][fno])
        tempf3 = []
        tempf3.append(filters[1][fno])
        for k in range(11):
            tempf1.append(filters[0][k][0][fno])
            tempf2.append(filters[0][k][1][fno])
            tempf3.append(filters[0][k][2][fno])
        w1,h1 = freqz(tempf1)
        w2,h2 = freqz(tempf2)
        w3,h3 = freqz(tempf3)
        plt.subplot(9,9,fno+1)
        plt.plot(w1,np.abs(h1),color='r')
        plt.plot(w2,np.abs(h2),color='g')
        plt.plot(w3,np.abs(h3),color='b')
plt.tight_layout()
plt.savefig('FilterMap.png')
print('Done.')

#######################################################

print('Saving model...',end='')
model.save('EqModel.hd5')
print('Done.')
