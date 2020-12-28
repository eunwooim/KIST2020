import numpy as np
import pandas as pd
import hypserspy.api as hs
from functions import escape_square_brackets
from computervision import matnorm255,clahe,erode,dm3toimg,

noise = np.random.normal(-5e-4,5e-4,(100,100))
ref = matnorm255(hs.load('/home/imeunu96/diffraction_patterns/Dataset2_Hour_00_Minute_00_Second_06_Frame_0001.dm3').data)
ref = clahe(ref.astype(np.uint8),cliplimit=5)
ref = ref + noise
ref = erode(ref)
ref = matnorm255(ref)

#Generating Dataset
docX = []
docY = []
f = pd.read_excel('/home/imeunu96/ML/code/savehere/lesseph.xlsx')
df = f.copy()
indexNames = df[ df['classified'] == 2 ].index
df.drop(indexNames , inplace=True)
df.to_excel('/home/imeunu96/ML/code/savehere/lesseph_without2.xlsx',index=False)
  #saved filepath and y data in 'lesseph.xlsx', if y=0, ring pattern. if y=1, spot pattern. if y=2, uncertain pattern.
  #training will proceed without uncertain patterns
f = pd.read_excel('/home/imeunu96/ML/code/savehere/lesseph_without2.xlsx')
for classified in f['classified']:
    for i in range(8):
        docY.append(classified)
  #since I augmented data with rotating by 8 times, save 8 times
print('Y_data Loaded')
# filename = np.array(x)
Y_data = np.array(docY)
for filepath in f['filepath']:
    try:
        print('loading %s' % (filepath))
        path = escape_square_brackets(filepath)
        img = dm3toimg(path,ref).astype(np.uint8)
        for angle in range(0,360,45):
            rotated = rotated_images(img,angle)
            docX.append(rotated)
    except:
        print('failed %s' %(filepath))
        nx = np.where(f['filepath'] == filepath)
        Y_data = np.delete(Y_data, nx[0][0])
X_data = np.array(docX)
docX.clear();docY.clear()
np.save('/home/imeunu96/ML/code/savehere/X_data.npy',X_data)
np.save('/home/imeunu96/ML/code/savehere/Y_data.npy',Y_data)
  #saved as .npy files
