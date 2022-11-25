# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.

def decaptcha( filenames ):
	# The use of a model file is just for sake of illustration
	with open( "model.txt", "r" ) as file:
		labels = file.read().splitlines()
	return labels