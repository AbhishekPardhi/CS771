# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.
import pickle
from processing import preprocess, mod_im
import cv2
import json

def decaptcha( filenames ):
	# The use of a model file is just for sake of illustration
	model = ''
	with open('model.pkl', 'rb') as file:
		model = pickle.load(file)
	
	label_map = json.load(open('label_map.json'))
	labels = []
	for file in filenames:
		img = cv2.imread(file)
		characters = preprocess(img)
		alphabets = []
		for char in characters:
			lab = model.predict(mod_im(char).flatten().reshape(1,-1))[0]
			alphabets.append(label_map[str(lab)])
		labels.append(','.join(alphabets))
	
	return labels


	