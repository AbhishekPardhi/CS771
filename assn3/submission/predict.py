# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.
import pickle
from processing import preprocess, features
import cv2
import json

def decaptcha( filenames ):
	# The use of a model file is just for sake of illustration

	# load the model
	model = ''
	with open('linear_model_binimg.pkl', 'rb') as file:
		model = pickle.load(file)
	
	# load the label map -> map from integer label to alphabets name
	label_map = json.load(open('label_map.json'))
	
	# processing(segmenting) the image and extracting features
	X = []
	for file in filenames:
		img = cv2.imread(file)
		characters = preprocess(img)
		for char in characters:
			X.append(features(char).flatten())
	
	# use model for getting predictions
	pred = model.predict(X)
	
	# prepare list of labels
	labels = []
	for i in range(len(filenames)):
		labels.append(
			label_map[str(pred[3*i])] + ',' +
			label_map[str(pred[3*i+1])] + ',' + 
			label_map[str(pred[3*i+2])] 
		)
	
	return labels


	