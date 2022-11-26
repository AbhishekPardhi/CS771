import cv2
import numpy as np
from scipy.signal import convolve2d

def remove_lines(img, bg, n=2):
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
    # img: image with lines removed
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

def preprocess(img, kernel_size = 2, freq_threshold = 400, pixel_threshold = 3):
    '''
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

def mod_im(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64,128))
    return gray