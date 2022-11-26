import cv2
import numpy as np
from scipy.signal import convolve2d

def remove_lines(img, bg, n=2):
    """
        remove noisy lines from the input image by edge detection and removing all edges.

        ARGUMENTS:
        img: image
        bg: background color of img
        n: size of kernel

        RETURNS:
        final_img: img of same shape as img with lines removed.
    """
    n = 2
    kernel = np.ones((2*n+1,2*n+1),np.uint32)
    img = img.astype(int)

    img2 = img.copy()
    for i in range(img.shape[2]):
        img2[:,:,i] = convolve2d(img[:,:,i], kernel, mode='same', boundary='wrap')

    numpix = (2*n+1)**2
    img3 = numpix*img

    final_img = np.where(img3 != img2, bg, img)
    return final_img

def get_freq(img):
    """
        compute frequency of each colours in the input image.

        ARGUMENTS:
        img: image

        RETURNS:
        freq: a dict with key color(a 3-tuple) and values their frequency in img.
    """

    freq = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color = tuple(img[i][j])
            if color in freq:
                freq[color] += 1
            else:
                freq[color] = 1
    return freq

def get_segmented(img, thr=400, th2=3):
    """
        segment images by computing histogram of non-background pixels into 3 seperate images each containing single character.

        ARGUMENTS:
        img: image with lines removed

        RETURNS:
        imgs: a list of images.
    """

    freq = get_freq(img)
    freq = {k:v for k,v in freq.items() if v>thr}
    freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)

    col1 = freq[1][0]
    col2 = ()
    col3 = ()
    if len(freq) == 2:
        # all characters are of same colour
        col2 = col1
        col3 = col1
    elif len(freq)==3:
        col2 = freq[2][0]
        col3 = col1
    else:
        col2 = freq[2][0]
        col3 = freq[3][0]

    nrow = img.shape[0]
    ncol = img.shape[1]
    
    hist = [0]*ncol
    for i in range(nrow):
        for j in range(ncol):
            if(tuple(img[i,j]) in [col1, col2, col3] ):
                hist[j] += 1

    imgs = []
    col = 0
    # print(hist)
    for i in range(3):
        start = 0
        stop = 0
        while hist[col]<th2+1:
            col+=1
        start = col
        col+=1
        while hist[col]>=th2 and col<ncol:
            col+=1
        stop = col
        col+=1
        # print(start,stop)
        imgs.append(img[:,start:stop+1,:])

    return imgs



def features(img, th2=3, ref_size = (64,64)):
    """
        Computes the feature vector of input img- a binary image with values of pixels belonging to character 1.

        ARGUMENTS:
        img: image with lines removed containing only one character
        th2: threshold for number of pixels in a row for the the row considered to be a part of alphabet.
        ref_size: the size of the output image. pls dont change this since the model is trained for 64,64.

        RETURNS:
        binimg: binary image.
    """
    freq = get_freq(img)
    freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    col1 = freq[1][0]
    
    row = 0
    start = 0
    end = 0
    num_nonbg = 0
    while num_nonbg < th2 +1:
        num_nonbg = 0
        for i in range(img.shape[1]):
            if(tuple(img[row,i]) == col1):
                num_nonbg+=1
        row+=1
    start = row-1
    
    row = img.shape[0]-1
    num_nonbg = 0
    while num_nonbg < th2 +1:
        num_nonbg = 0
        for i in range(img.shape[1]):
            if(tuple(img[row,i]) == col1):
                num_nonbg+=1
        row-=1
    end = row+1
    
    img = img[start:end+1,:,:]
    binimg = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8) 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(tuple(img[i,j]) == col1):
                binimg[i,j] = 255

    binimg = cv2.resize(binimg, ref_size)
    return binimg


def preprocess(img, kernel_size = 2, freq_threshold = 400, pixel_threshold = 3):
    '''
        preprocess the image by segmenting it.

        ARGUMENTS:
        img: input image.
        kernel size: for convolution used for erosion.
        freq_threshold: to ignore colors of frequency less than this.
        pixel_therhold: ignore cols that contains less non-bg pixels than this.

        # RETURNS:
        imgs: a list of 3 images each containing a single character
    '''

    freq_org = get_freq(img)
    bg_color = max(zip(freq_org.values(), freq_org.keys()))[1]
    
    lines_removed = remove_lines(img, bg_color, kernel_size)

    return get_segmented(lines_removed, freq_threshold, pixel_threshold)
