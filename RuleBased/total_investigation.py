import os
import cv2
from computervision import walk, isdm3, predcir

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
print(ring, spot, err)
