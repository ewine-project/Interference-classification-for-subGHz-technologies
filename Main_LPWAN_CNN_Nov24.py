import keras
import os,random
import pickle
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"]  = "device=gpu%d"%(1)
import keras
from keras.models import Sequential
import scipy.io as sio
import array 
import numpy as np
from matplotlib import pyplot as plt
from keras import losses
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.optimizers import RMSprop
from IPython.display import clear_output
from keras import optimizers
from keras.models import load_model

numclass=10
in_dim=[2,500]
snrs=[-70,-60,-50,-40,-30,-20,-10,0,10,20]
#loading sensing-shots
# For training -------------------------------------------------------------------------
load_data = sio.loadmat('testing_training_combined_fft.mat')
X = load_data['X_combined']
X_label=load_data['X_combined_label']
lbl=load_data['label']
X_labeld = keras.utils.to_categorical(X_label,numclass)
[r_comb,c_comb]=X.shape
X=np.reshape(X,(int(r_comb),2,int(c_comb/2))) 

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.6
train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
Y_train_labeld=X_labeld[train_idx]
Y_test_labeld=X_labeld[test_idx]

#train_idx=range(0,104000)
#test_idx=range(104000,200000)
#X_train = X[train_idx]
#X_test =  X[test_idx]
#Y_train_labeld=X_labeld[train_idx]
#Y_test_labeld=X_labeld[test_idx]

#keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

dr = 0.0

model = Sequential()
model.add(Reshape(in_dim+[1],input_shape=in_dim))
#model.add(ZeroPadding2D((2, 2)))
model.add(Conv2D(64, (2, 3), name='conv1', padding='valid', activation='relu', kernel_initializer='glorot_uniform'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(dr))
#model.add(ZeroPadding2D((2, 2)))
model.add(Conv2D(16, (1, 3), name='conv2', padding='valid', activation='relu', kernel_initializer='glorot_uniform'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(128, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(numclass, init='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([numclass]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
#
nb_epoch = 5
batch_size = 400
filepath = 'LPWANrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train, Y_train_labeld,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=1,
                    validation_split=0.2,
                    validation_data=(X_test, Y_test_labeld),callbacks=[plot_losses])

#model.load_weights(filepath)

score = model.evaluate(X_test, Y_test_labeld, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('LPWAN_epochs_200_batchsize_400_dropout_0.0_classes_8_Split_60n40_with_fft.h5') 

# Show loss curves 
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
#
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#    
## Plot confusion matrix
classes=['Sigfox - ch0', 'Sigfox - ch180', 'Sigfox - ch400', 'IEEE 802.15.4 (subGHz)', 'LoRA-SF7','LoRA-SF8','LoRA-SF9','LoRA-SF10','LoRA-SF11','LoRA-SF12']
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm1 = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test_labeld[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm1[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm1, labels=classes)
    
# Plot confusion matrix
acc = {}
conf_tot={}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    temp=np.where(np.array(test_SNRs)==snr)
    test_X_i = X_test[temp[0]]
    test_Y_i = Y_test_labeld[temp[0]]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    conf_tot[snr]=confnorm  
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)


# saving variables
f_myfile = open('LPWAN_epochs_200_batchsize_400_dropout_0.0_classes_8_Split_60n40_with_fft.pickle', 'wb')
pickle_objects={'snr':snrs,'confusion_mart_snrs':conf_tot,'confusion_matrix_SNR':confnorm1,'classification_acc_with_SNR':acc, 'training_loss':history.history['loss'],'validation_loss':history.history['val_loss'],'validation_accuracy':history.history['val_acc'],'training_accuracy':history.history['acc']}
pickle.dump(pickle_objects, f_myfile)
f_myfile.close()

# 
# file = open('LPWAN_epochs_200_batchsize_400_dropout_0.0_classes_8_Split_60n40_with_fft.pickle', 'rb')
# object = pickle.load(file)
    
    