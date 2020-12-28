import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

def matnorm255(matrix,size=500,v=None):
    '''Convert raw matrix into uint8 image matrix with Adjusted Contrast'''
    if v is None:
        v = 1
    x = abs(matrix)
    v = x.max()/v
    x[x>v]=v
    gray = 255*(x-x.min())/(x.max()-x.min())
    gray = gray.astype(np.float32)
    gray = cv2.resize(gray,dsize=(size,size), interpolation=cv2.INTER_AREA)
    return gray

def fft(matrix, scale=None, value=None):
    '''Generate FFT Image'''
    img_c2 = np.fft.fft2(matrix)
    img_c2 = np.fft.fftshift(img_c2)
#     img_c4 = np.fft.ifftshift(img_c3)
#     img_c5 = np.fft.ifft2(img_c4)
    if value is None:
        img_c4 = abs(img_c2)
    elif value == 'real':
        img_c4 = img_c2.real
#     elif value == 'imag':
#         img_c4 = img_c4.imag
    else:
        raise IndexError("Choose in 'abs' or 'real'")
    img_c4 = cv2.resize(img_c4,dsize=(800, 800), interpolation=cv2.INTER_AREA)
    if scale is None:
        img = np.log(1+img_c4)
    elif scale == 'sqrt':
        img = np.sqrt(1+img_c4)
    else:
        raise IndexError("Choose in 'log' or 'sqrt'")
    return img

def morphopen(img,kersize=3,iteration=1):
    '''Opening Morphology'''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kersize,kersize))
    dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iteration)
    return dst

def binary(img,thres=50):
    '''Convert gray image into Binary image'''
    ret, binary = cv2.threshold(img, thres, 100, cv2.THRESH_BINARY)
    return binary

def erode(img,kersize=3,iteration=2):
    '''Morphology Erosion'''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kersize,kersize))
    erode = cv2.erode(img, kernel, anchor=(-1, -1), iterations=iteration)
    return erode

def hcircle(img,mindist=1200,par1=390,par2=15,minr=90,maxr=450,show=False):
    '''Find Circles with Hough Transformation'''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1200, param1 = par1,
                               param2 = par2, minRadius = minr, maxRadius = maxr)
    if circles is None:
#         print('No circle')
        return [img,0]
    else:
        #print('Circle detected')
        if show == True:
            src = img.copy()
            for i in circles[0]:
                cv2.circle(src, (i[0], i[1]), int(i[2]), (0, 0, 255), 1)
            cv2.imshow('Circle Detected',src)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return [src,1]
        return [img,1]

def hline(img,par=10,show=False):
    '''Find Lines with Hough Transformation'''
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,par)
    if lines is None:
#         print('No line')
        return [img,0]
    else:
#         print(str(len(lines))+' lines Detected')
        if show == True:
            for rho,theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.imshow('Line Detected',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return [img,1]

def clahe(img,cliplimit=2):
    '''Apply Contrast Limited Adaptive Histogram Equalization'''
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8,8))
    claheimg = clahe.apply(img)
    return claheimg

def matchhist(img,ref):
    '''Apply Histogram Matching to an image, Target : ref'''
    matched = match_histograms(img,ref,multichannel=False)
    return matched

def predcir(name):
    '''
    Predict whether the dm3 image has circle
    if circle exist, return circled image and 1
    if not, return preprocessed image and 0
    '''
    x = loaddm3(name)
    img = matnorm255(x)
    img1 = matchhist(img,ref)
    img2 = img1.astype(np.uint8)
    img3 = erode(img2)
    img4 = binary(img3)
    circles = cv2.HoughCircles(img4, cv2.HOUGH_GRADIENT, 1, 1200, param1 = 390,
                               param2 = 15, minRadius = 30, maxRadius = 190)
    src = img2.copy()
    if circles is None:
        return [img3,0]
    else:
        src = img3.copy()
        for i in circles[0]:
            cv2.circle(src, (i[0], i[1]), int(i[2]), (0, 0, 255), 1)
        return [src,1]
    
def imshow(img):
    '''Pop up the image'''
    img = img.astype(np.uint8)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def gaussianSmoothing(img,kersize=5,sig=2):
    '''Apply Gaussian Smoothing'''
    kernel = cv2.getGaussianKernel(kersize,sig)
    ker = np.outer(kernel, kernel.transpose())
    img = cv2.filter2D(img, -1, ker)
    return img

def highfreqimg(img,kersize=5,sig=2):
    '''Return High Frequency Image by Subtracting Original Image and Smoothed Image'''
    blur = gaussianSmoothing(img,kersize,sig)
    high = img - blur + 128
    return high

def dm3toimg(filepath,ref):
    '''Load .dm3 data and adjust contrast, return image'''
    x = hs.load(filepath).data
    img = matnorm255(x,len(ref))
    img = matchhist(img,ref)
    return img

def fit_line_ransac(data,iter=30,sample_num=10,offset=80.0):
	'''Fit line with RANSAC algorithm'''
	count_max = 0
	  effective_sample = None
	  for i in range(iter):
		  sample = np.random.choice(len(data), sample_num, replace=False)
		    xs = data[sample][:,0].reshape(-1,1)
		    ys = data[sample][:,1].reshape(-1,1)
		J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
		Y = np.mat(-1*xs**2)
		P= (J.T * J).I * J.T * Y
		# fitter a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
		a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
		ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
		# threshold 
		    ran_sample = np.array([[x,y] for (x,y) in data if np.abs(ellipse_model(x,y)) < offset ])
		if(len(ran_sample) > count_max):
			count_max = len(ran_sample) 
			      effective_sample = ran_sample
	return fit_rotated_ellipse(effective_sample)
    
def rotate_images(img, angle):
    '''Rotate image with input angle'''
    (h, w) = img.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated
