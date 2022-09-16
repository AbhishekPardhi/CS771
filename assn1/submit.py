import numpy as np
# This is the only scipy method you are allowed to use
# Use of scipy is not allowed otherwise
from scipy.linalg import khatri_rao
import random as rnd
import time as tm





def stepLengthGenerator( mode, eta ):
	if mode == "constant":
		return lambda t: eta
	elif mode == "linear":
		return lambda t: eta/(t+1)
	elif mode == "quadratic":
		return lambda t: eta/np.sqrt(t+1)

class SVM:
    def __init__(self, X, y, C):
        self.X = X.copy()
        self.y = y.copy()
        self.C = C
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.__intialize_params__()

    def __intialize_params__(self):
        X = self.X
        y = self.y

		# param intiliaziation for SDCM
		
        initDual = self.C * np.ones((self.n))
        self.normSq = np.square( np.linalg.norm( X, axis = 1 ) ) + 1
        self.w_SDCM = X.T.dot( np.multiply( initDual, y ) )
        self.alpha = initDual

		# param init for GD
        self.theta = np.zeros((self.d))
    
    def predict(self, X):

        y_new = X.dot(self.w_SDCM) 
        return y_new

    def doCoordOptCSVMDual(self, i ):
        x = self.X[i,:]
        y = self.y
        w_SDCM = self.w_SDCM
        C = self.C

        # Find the unconstrained new optimal value of alpha_i
        # It takes only O(d) time to do so because of our clever book keeping
        newAlphai = (1 - y[i] * x.dot(w_SDCM) ) / self.normSq[i]
        
        # Make sure that the constraints are satisfied. This takes only O(1) time
        if newAlphai > C:
            newAlphai = C
        if newAlphai < 0:
            newAlphai = 0

        # Update the primal model vector and bias values to ensure bookkeeping is proper
        # Doing these bookkeeping updates also takes only O(d) time
        self.w_SDCM = w_SDCM + (newAlphai - self.alpha[i]) * y[i] * x
        
        return newAlphai

    def getCSVMPrimalDualObjVals( self ):
        w_SDCM = self.w_SDCM
        X = self.X
        y = self.y
        C = self.C

        hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w_SDCM )), y ), 0 )
        objPrimal = 0.5 * w_SDCM.dot( w_SDCM ) + C * np.sum(hingeLoss)
        # Recall that b is supposed to be treated as the last coordinate of w
        objDual = np.sum( self.alpha ) - 0.5 * np.square( np.linalg.norm( w_SDCM ) )
        
        return np.array( [objPrimal, objDual] )

    def getRandpermCoord( self, state ):
        idx = state[0]
        perm = state[1]
        d = len( perm )
        if idx >= d - 1 or idx < 0:
            idx = 0
            perm = np.random.permutation( d )
        else:
            idx += 1
        state = (idx, perm)
        curr = perm[idx]
        return (curr, state)
    
    def doSDCM(self, horizon = 10 ):
        objValSeries = []
        timeSeries = []
        totTime = 0
        d = self.d
        state = (-1,np.random.permutation( d ))
        # Initialize model as well as some bookkeeping variables
        
        for it in range( horizon ):
            # Start a stopwatch to calculate how much time we are spending
            tic = tm.perf_counter()
            
            # Get the next coordinate to update and update that coordinate
            (i, state) = self.getRandpermCoord( state )
            (self.alpha)[i] = self.doCoordOptCSVMDual(  i)

            toc = tm.perf_counter()
            totTime = totTime + (toc - tic)
            # print('\r Accuracy:', self.eval(self.X, self.y), end='')
            objValSeries.append( self.getCSVMPrimalDualObjVals() )
            timeSeries.append( totTime )
            
        return (objValSeries, timeSeries)

    def getCSVMGrad(self, theta ):
        y = self.y
        C = self.C
        X = self.X

        w = theta
        discriminant = np.multiply( (X.dot( w )), y )
        g = np.zeros( (y.size,) )
        g[discriminant < 1] = -1
        # delb = C * g.dot( y )
        delw = w + C * (X.T * g).dot( y )
        return delw

    def getCSVMObjVal(self,  theta ):
        C = self.C
        X = self.X
        y = self.y

        w = theta
        hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w )), y ), 0 )
        return 0.5 * w.dot( w ) + C * np.sum( hingeLoss )
    
    def clean_up(self, cumulative, doModelAveraging, it ):
        final = 0
        if doModelAveraging:
            final = cumulative / (it + 1)
        else:
            final = cumulative
        
        self.theta = final
        self.w_SDCM = final
	
    def doGD( self, stepFunc , doModelAveraging, spacing, timeout):
        objValSeries = []
        timeSeries = []
        totTime = 0
        theta = self.theta
        cumulative = self.theta
        it = 1
        
        while True:
            # Start a stopwatch to calculate how much time we are spending
            tic = tm.perf_counter()
            delta = self.getCSVMGrad( theta )
            theta = theta - stepFunc( it + 1 ) * delta
            # If we are going to do model averaging, just keep adding the models
            if doModelAveraging:
                cumulative = cumulative + theta
            else:
                cumulative = theta
            # All calculations done -- stop the stopwatch
            toc = tm.perf_counter()
            totTime = totTime + (toc - tic)
            # If model averaging is being done, need to calculate current objective value a bit differently
            if doModelAveraging:
                objValSeries.append( self.getCSVMObjVal( cumulative/(it+2) ) )
            else:
                objValSeries.append( self.getCSVMObjVal( cumulative ) )

            timeSeries.append( totTime )
            
            if it%spacing ==0 and totTime > timeout:
                self.clean_up(cumulative, doModelAveraging, it)
                return (objValSeries, timeSeries)
            it+=1


    def eval(self, X_t, y_t):
        y_t_pred = self.predict( X_t )
        y_t_pred = np.where( y_t_pred > 0, 1, -1 )
        acc = np.average( y_t == y_t_pred )
        return acc





# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES FOR WHATEVER REASON WILL RESULT IN A STRAIGHT ZERO
# THIS IS BECAUSE THESE PACKAGES CONTAIN SOLVERS WHICH MAKE THIS ASSIGNMENT TRIVIAL
# THE ONLY EXCEPTION TO THIS IS THE USE OF THE KHATRI-RAO PRODUCT METHOD FROM THE SCIPY LIBRARY
# HOWEVER, NOTE THAT NO OTHER SCIPY METHOD MAY BE USED IN YOUR CODE

# DO NOT CHANGE THE NAME OF THE METHODS solver, get_features, get_renamed_labels BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def get_renamed_labels( y ):
################################
#  Non Editable Region Ending  #
################################

	# Since the dataset contain 0/1 labels and SVMs prefer -1/+1 labels,
	# Decide here how you want to rename the labels
	# For example, you may map 1 -> 1 and 0 -> -1 or else you may want to go with 1 -> -1 and 0 -> 1
	# Use whatever convention you seem fit but use the same mapping throughout your code
	# If you use one mapping for train and another for test, you will get poor accuracy
	
	# replace 0 in y with -1
	y[y==0] = -1
	return y
	# return y_new.reshape( ( y_new.size, ) )					# Reshape y_new as a vector


################################
# Non Editable Region Starting #
################################
def get_features( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this function to transform your input features (that are 0/1 valued)
	# into new features that can be fed into a linear model to solve the problem
	# Your new features may have a different dimensionality than the input features
	# For example, in this application, X will be 8 dimensional but your new
	# features can be 2 dimensional, 10 dimensional, 1000 dimensional, 123456 dimensional etc
	# Keep in mind that the more dimensions you use, the slower will be your solver too
	# so use only as many dimensions as are absolutely required to solve the problem

	# let n = number of training examples.

	# X is n * 8 matrix
	D = np.flip(2*X-1,axis = 1)
	X_inter = np.cumprod(D, axis = 1)

	X_inter = np.insert(X_inter, 0, 1,axis = 1)			# X_inter is n*9 mat

	X_inter = X_inter.T								# X_inter is 9*n mat

	X_new = khatri_rao(X_inter, X_inter)			# X_new is khatri_rao product, shape = 9^2*n 
	X_new = khatri_rao(X_new,X_inter )				# X_new.shape = 9^3*n 
	X_new = X_new.T
	return X_new


################################
# Non Editable Region Starting #
################################
def solver( X, y, timeout, spacing ):
	
	# W is the model vector and will get returned once timeout happens
	# B is the bias term that will get returned once timeout happens
	# The bias term is optional. If you feel you do not need a bias term at all, just keep it set to 0
	# However, if you do end up using a bias term, you are allowed to internally use a model vector
	# that hides the bias inside the model vector e.g. by defining a new variable such as
	# W_extended = np.concatenate( ( W, [B] ) )
	# However, you must maintain W and B variables separately as well so that they can get
	# returned when timeout happens. Take care to update W, B whenever you update your W_extended
	# variable otherwise you will get wrong results.
	# Also note that the dimensionality of W may be larger or smaller than 9


	(n, d) = X.shape
	t = 0
	totTime = 0
	W = []
	B = 0
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

	# You may reinitialize W, B to your liking here e.g. set W to its correct dimensionality
	# You may also define new variables here e.g. step_length, mini-batch size etc
	def stepLengthGenerator( mode, eta ):
		if mode == "constant":
			return lambda t: eta
		elif mode == "liear":
			return lambda t: eta/(t+1)
		elif mode == "quadratic":
			return lambda t: eta/np.sqrt(t+1)

	theta = np.zeros(d)
	
	C=5.0
	
	
	def getCSVMObjVal( theta ):
		w = theta
		hingeLoss = np.maximum( 0, 1 - y * ( np.dot( X, w ) ) )
		return 0.5 * np.dot( w, w ) + C * np.sum( hingeLoss )

	def getCSVMGrad( theta ):
		w = theta	
		discriminant = y * ( np.dot( X, w ) )
		g = np.zeros( (y.size,))
		g[discriminant < 1] = -1
		delw = w + C * np.dot( X.T * g, y )
		return delw

	# def clean_up(cumulative, doModelAveraging, it):
	# 	final = 0
	# 	if doModelAveraging:
	# 		final = cumulative/(it+1)
	# 	else:
	# 		final = cumulative
	# 	theta=final
	
	objValSeries = []
	timeSeries = []
	cumulative = theta
	
		

################################
# Non Editable Region Starting #
################################

	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return ( W.reshape( ( W.size, ) ), B, totTime )			# Reshape W as a vector
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses which will be strictly penalized
		
		# Note that most likely, you should be using get_features( X ) and get_renamed_labels( y )
		# in this part of the code instead of X and y -- please take care
		
		# Please note that once timeout is reached, the code will simply return W, B
		# Thus, if you wish to return the average model (as is sometimes done for GD),
		# you need to make sure that W, B store the averages at all times
		# One way to do so is to define a "running" variable w_run, b_run
		# Make all GD updates to W_run e.g. W_run = W_run - step * delW (similarly for B_run)
		# Then use a running average formula to update W (similarly for B)
		# W = (W * (t-1) + W_run)/t
		# This way, W, B will always store the averages and can be returned at any time
		# In this scheme, W, B play the role of the "cumulative" variables in the course module optLib (see the cs771 library)
		# W_run, B_run on the other hand, play the role of the "theta" variable in the course module optLib (see the cs771 library)

		delta = getCSVMGrad( theta )
		theta = theta - stepLengthGenerator( "linear", 1 )( t + 1 ) * delta
		cumulative = cumulative + theta
		objValSeries.append( getCSVMObjVal( cumulative/(t+2) ) )
		timeSeries.append( totTime )

	
		
	return ( W.reshape( ( W.size, ) ), B, totTime )			# This return statement will never be reached