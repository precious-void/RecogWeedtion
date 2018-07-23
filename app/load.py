import keras.models
from keras.models import load_model
import tensorflow as tf
import keras
from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger

# Initialize CNN model of classification stages of growth
def init_neural_network(path): 
	clear_session()
	loaded_model = load_model(path)
	graph = tf.get_default_graph()
	return loaded_model, graph

# Initialize CNN model of recognizing the crop
def init_recognition_cnn(scale = 70):
	graph = tf.get_default_graph()
	
	model = Sequential()
	model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))
	model.add(BatchNormalization(axis=3))
	model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(BatchNormalization(axis=3))
	model.add(Dropout(0.1))
	model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
	model.add(BatchNormalization(axis=3))
	model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(BatchNormalization(axis=3))
	model.add(Dropout(0.1))
	model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
	model.add(BatchNormalization(axis=3))
	model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(BatchNormalization(axis=3))
	model.add(Dropout(0.1))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(12, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Loading weights
	model.load_weights("CNN/weights_best_17-0_96.hdf5")
	return model, graph
