#Labelling
import os
from functions import walk, loaddm3, 
from computervision import matnorm255, matchhist, imshow

destdir = '/home/imeunu96/diffraction_patterns/Titan/'
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
