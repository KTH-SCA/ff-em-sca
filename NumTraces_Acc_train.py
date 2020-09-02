import os
import os.path
import sys
import h5py
import numpy as np
import scipy.io as scio
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, \
    AveragePooling1D,Dropout, LSTM
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.preprocessing import PolynomialFeatures



def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def lstm_best(classes=256):
    model = Sequential()
    model.add(LSTM(10,input_shape=(7, 1),return_sequences=True ))
    model.add(LSTM(10))
    model.add(Dense(classes, activation='softmax'))
	
	
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	

	
    return model
	
	

	
	
	
#### MLP Best model  
def mlp_best(classes=256):  


    model = Sequential()
	
    model.add(Dense(100, input_dim=10, activation='relu'))  #input_dim=5  all are 100 nodes before
	
    model.add(Dense(100, activation='relu'))
	
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
		
    model.add(Dense(classes, activation='softmax'))
	
	
    optimizer = Adam(learning_rate=0.0001)   
	
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	

	
    return model


### CNN Best model 
def cnn_best(classes=256):  

    '''
	#35 epoch ,128 batch_size, 7 points
    model = Sequential()
    model.add(Conv1D(input_shape=(5, 1) , filters=5, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1))
    
    model.add(Conv1D(input_shape=(5, 1) , filters=10, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1)) 
    
    model.add(Flatten())
	
    #model.add(Dropout(0.2))
	
    model.add(Dense(units = 200, activation = 'relu'))
    model.add(Dense(units = 200, activation = 'relu'))
    
    model.add(Dense(units = classes, activation = 'softmax',name='predictions'))
	
    optimizer = RMSprop(lr=0.0001)
	
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    '''

	

   
	# 70-100 epoch ,128 batch_size, 110 points
    model = Sequential()
    model.add(Conv1D(input_shape=(110, 1) , filters=4, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1))   	
	
    model.add(Conv1D( filters=8, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1)) 

    model.add(Conv1D( filters=16, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1)) 	

    model.add(Conv1D( filters=32, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1)) 
	
    model.add(Flatten())
	
    #model.add(Dropout(0.2))
	
    model.add(Dense(units = 200, activation = 'relu'))
    model.add(Dense(units = 200, activation = 'relu'))
	
    model.add(Dense(units = classes, activation = 'softmax',name='predictions'))	
    optimizer = RMSprop(lr=0.00005) #0.0001
	
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])	
    

	
    '''
    model = Sequential()
    model.add(Conv1D(input_shape=(10, 1) , filters=4, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1))   	
	
    model.add(Conv1D( filters=8, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1)) 

    model.add(Conv1D( filters=16, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1)) 	

    model.add(Conv1D( filters=32, kernel_size=3, activation='relu', padding='same'))	
    model.add(AveragePooling1D(pool_size=2,strides=1)) 
	
    model.add(Flatten())
	
    #model.add(Dropout(0.2))
	
    model.add(Dense(units = 200, activation = 'relu'))
    model.add(Dense(units = 200, activation = 'relu'))
	
    model.add(Dense(units = classes, activation = 'softmax',name='predictions'))	
    optimizer = RMSprop(lr=0.0001)
	
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])	
    '''
	
    return model




#### Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs, batch_size):
    check_file_exists(os.path.dirname(save_file_name))
    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)
    callbacks = [save_model]
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    # Sanity check
    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (
            input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        # This is a CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)


    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256), batch_size=batch_size,
                        verbose=1, epochs=epochs, callbacks=callbacks, validation_split=0.1)
    return history



if __name__ == "__main__":


    NUMBER = 500000

    Traces = np.load('nor_traces_maxmin.npy' )
	
	
	
	
    Traces = Traces[:NUMBER]

    #Traces = Traces[:,[60,61,62,63, 132, 133,134,135,136,137]]   #132,137  #130,240   # 50, 380  [i for i in range(132,137)]
    Traces = Traces[:,[i for i in range(130,240)]]   #132,137  #130,240    # 50, 380


    labels = np.load('label_0.npy' )
    labels = labels[:NUMBER]


    model_folder = 'models_cnn_5points/'


    #training_model = model_folder + 'model_mlp-{epoch:01d}.h5'
    training_model = model_folder + 'model_cnn-{epoch:01d}.h5'
    #training_model =  'model_lstm.h5'
	
    epochs = 100
    batch_size = 128



    # get network type

    #best_model = mlp_best()
    best_model = cnn_best()
    #best_model = lstm_best()


    ### training


    history_log = train_model(Traces, labels, best_model, training_model, epochs, batch_size)

    print(history_log.history['val_accuracy'])

    acc = np.array(history_log.history['val_accuracy'])
    loss = np.array(history_log.history['val_loss'])
    
    np.save(model_folder + 'Acc.npy', acc)
    np.save(model_folder + 'Loss.npy', loss)
