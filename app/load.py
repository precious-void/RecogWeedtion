import keras.models
from keras.models import load_model
import tensorflow as tf
import keras

def init(): 
	loaded_model = load_model("model.h5")
	# load woeights into new model
	# loaded_model.load_weights("model.h5")
	# print("Loaded Model from disk")
	# compile and evaluate loaded model

	graph = tf.get_default_graph()

	return loaded_model, graph