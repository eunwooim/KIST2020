import os
import re
import cv2
import csv
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import hyperspy.api as hs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.exposure import match_histograms

def isdm3(ext):
  dmlist = ['.dm3', '.dm4']
  if ext in dmlist:
    return True
  else:
    return False

def walk(folder):
  """Walk through every files in a directory"""
  for dirpath, dirs, files in os.walk(folder):
    for filename in files:
      yield dirpath, filename

def kth(arr, K):
  '''Find Kth largest number in array'''
  ORDINAL_MSG = ('st', 'nd', 'rd')[K-1] if K <= 3 else 'th'
  unique_set = set(arr)
  if len(unique_set) < K:
    raise IndexError(f"There's no {K}{ORDINAL_MSG} value in array")
  elif K <= 0 or not arr:
    raise ValueError("K should be over 0 and arr should not be empty")
  INF = float('inf')
  memory = [-INF] * K
  for n in arr:
    if n <= memory[-1]:
      continue
    for i, m in enumerate(memory):
      if (i == 0 and n > m) or m < n < memory[i-1]:
        for j in range(len(memory)-2, i-1, -1):
          memory[j+1] = memory[j]
        memory[i] = n
        break
  return memory[-1]

def readyplot():
  plt.figure(figsize=(8,8))
  plt.tight_layout()
  plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
  plt.axis('off'), plt.xticks([]), plt.yticks([])

def loaddm3(name):
  '''Load dm3 or dm4 raw matrix'''
  if name[-4:] in ['.dm3','.dm4']:
    matrix = hs.load(name).data
  else:
    try:
      newname = name+'.dm3'
      matrix = hs.load(newname).data
    except:
      newname = name+'.dm4'
      matrix = hs.load(newname).data
  return matrix

def matnorm255(matrix, v=None):
  '''Convert raw matrix into uint8 image matrix with Adjusted Contrast'''
  if v is None:
    v = 1
  x = abs(matrix)
  v = x.max()/v
  x[x>v]=v
  gray = 255*(x-x.min())/(x.max()-x.min())
  gray = gray.astype(np.float32)
  gray = cv2.resize(gray,dsize=(800, 800), interpolation=cv2.INTER_AREA)
  return gray

def fft(matrix, scale=None, value=None):
  '''Generate FFT Image'''
  img_c2 = np.fft.fft2(matrix)
  img_c2 = np.fft.fftshift(img_c2)
#   img_c4 = np.fft.ifftshift(img_c3)
#   img_c5 = np.fft.ifft2(img_c4)
  if value is None:
    img_c4 = abs(img_c2)
  elif value == 'real':
    img_c4 = img_c2.real
#  elif value == 'imag':
#    img_c4 = img_c4.imag
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

def morphopening(img,kersize=3,iteration=1):
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
  circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1200, param1 = par1, param2 = par2, minRadius = minr, maxRadius = maxr)
  if circles is None:
    # print('No circle')
    return [img,0]
  else:
#    print('Circle detected')
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
    # print('No line')
    return [img,0]
  else:
    # print(str(len(lines))+' lines Detected')
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
  x = loaddm3(name)
  img = matnorm255(x)
  img1 = matchhist(img,ref)
  img2 = img1.astype(np.uint8)
  img3 = erode(img2)
  img4 = binary(img3)
  circles = cv2.HoughCircles(img4, cv2.HOUGH_GRADIENT, 1, 1200, param1 = 390, param2 = 15, minRadius = 30, maxRadius = 190)
  src = img2.copy()
  if circles is None:
    return [img3,0]
  else:
    src = img3.copy()
    for i in circles[0]:
      cv2.circle(src, (i[0], i[1]), int(i[2]), (0, 0, 255), 1)
    return [src,1]
    
def imshow(img):
  img = img.astype(np.uint8)
  cv2.imshow('img',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return

def gaussianSmoothing(img,kersize=5,sig=2):
  kernel = cv2.getGaussianKernel(kersize,sig)
  ker = np.outer(kernel, kernel.transpose())
  img = cv2.filter2D(img, -1, ker)
  return img

def highfreqimg(img,kersize=5,sig=2):
  blur = gaussianSmoothing(img,kersize,sig)
  high = img - blur + 128
  return high

def dm3toimg(img,ref):
  x = hs.load(filepath).data
  img = matnorm255(x)
  img = matchhist(img,ref)
  return img

walkdir = '/mnt/TEM/Titan/2 Researcher/'
destdir = '/home/imeunu96/diffraction_patterns/Titan/JOSEPH'
count = 0

noise = np.random.normal(-5e-4,5e-4,(800,800))
ref = matnorm255(loaddm3('/home/imeunu96/diffraction_patterns/Dataset2_Hour_00_Minute_00_Second_06_Frame_0001'))
ref = clahe(ref.astype(np.uint8),cliplimit=5)
ref = ref + noise
ref = erode(ref)
ref = matnorm255(ref)

'''
#Classify
ring=spot=error=count=0
for folder, filename in walk(destdir):
    ext = os.path.splitext(filename)[-1]
    if not isdm3(ext):
        continue
    filepath = os.path.join(folder, filename).replace("\\","/")
    paths = os.path.normpath(filepath).split(os.sep)
    newdir = os.path.join(destdir,'/'.join(paths[5:-1]))
    #os.makedirs(newdir, exist_ok=True)
    #shutil.copy2(filepath, newdir)
    count += 1
    if count>=1000:
        break
    try:
        if predcir(filepath)[1]==1:
            os.chdir(ringdir)
            print("# %s [%d]" % (filepath, count))
            cv2.imwrite(str(count)+'.png',predcir(filepath)[0])
            ring += 1
        else:
            os.chdir(spotdir)
            print("%s [%d]" % (filepath, count))
            cv2.imwrite(str(count)+'.png',predcir(filepath)[0])
            spot += 1
    except:
        print("error %s [%d]" % (filepath,count))
        err += 1

#Confusion Matrix
array = [[620,5],
        [344,20]]
df_cm = pd.DataFrame(array, index = [i for i in ['Ring','Spot']], columns = [i for i in ['Ring','Spot']])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm,cmap='YlGnBu',annot=True,fmt='d')

#Labelling
f = open('classifier.csv','w')
count=0
for folder, filename in walk(destdir):
  ext = os.path.splitext(filename)[-1]
  if not isdm3(ext):
    continue
  filepath = os.path.join(folder, filename).replace("\\","/")
  x = loaddm3(filepath)
  img = matnorm255(x)
  img = matchhist(img,ref)
  imshow(img)
  i = input()
  f.write(filepath + ',' + str(i) + '\n')
f.close()
'''
