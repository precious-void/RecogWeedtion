import keras.models
from keras.models import load_model
import tensorflow as tf
import keras
from keras.backend import clear_session

#Initialize CNN model
def init_neural_network(path): 
	clear_session()
	loaded_model = load_model(path)
	graph = tf.get_default_graph()
	return loaded_model, graph