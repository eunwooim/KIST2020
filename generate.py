import os
import cv2
import csv
import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

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

def clahe(img,cliplimit=2):
      '''Apply Contrast Limited Adaptive Histogram Equalization'''
  clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8,8))
  claheimg = clahe.apply(img)
  return claheimg

def matchhist(img,ref):
      '''Apply Histogram Matching to an image, Target : ref'''
  matched = match_histograms(img,ref,multichannel=False)
  return matched

def dm3toimg(img,ref):
  x = hs.load(img).data
  img = matnorm255(x)
  img = matchhist(img,ref)
  return img

noise = np.random.normal(-5e-4,5e-4,(800,800))
ref = matnorm255(loaddm3('/home/imeunu96/diffraction_patterns/Dataset2_Hour_00_Minute_00_Second_06_Frame_0001'))
ref = clahe(ref.astype(np.uint8),cliplimit=5)
ref = ref + noise
ref = erode(ref)
ref = matnorm255(ref)

os.chdir('/home/imeunu96/')
f = open('testing.csv','r',encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
  #x = 
f.close()
#img_dir = 
categories = ['ring','spot']
nb_classes = len(categories)