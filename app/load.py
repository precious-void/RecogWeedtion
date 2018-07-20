import keras.models
from keras.models import model_from_json
import tensorflow as tf
import keras

def init(): 
	json_file = open('file.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load woeights into new model
	loaded_model.load_weights("model.h5")
	# print("Loaded Model from disk")
	# compile and evaluate loaded model

	graph = tf.get_default_graph()

	return loaded_model, graph