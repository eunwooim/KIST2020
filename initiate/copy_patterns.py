import os
import re
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
plt.ion()

def isdiff(signal):
    if not isinstance(signal, list):
      signals = [signal]
    else:
        signals = signal
    for signal in signals:
        axes = signal.axes_manager.as_dictionary()
        count = 0
        for axis in axes:
            if axes[axis]['units']=="1/nm":
                count += 1
        if count>=2:
            return True
    return False

def isdm3(ext):
    dmlist = ['.dm3', '.dm4']
    if ext in dmlist:
        return True
    else:
        return False

def escape_square_brackets(text):
    rep = dict((re.escape(k), v) for k, v in {"[": "[[]", "]": "[]]"}.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

prev_dirs = ['''Write Previous Directories''']

def walk(folder):
  """Walk through every files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        paths = os.path.normpath(dirpath).split(os.sep)
        if len(paths)>5 and paths[5] in prev_dirs:
            continue
        for filename in files:
            yield dirpath, filename

# walkdir = 'D:/Programs/machine_learning/diffraction_machinelearning/experiments/'
walkdir = '/mnt/TEM/Titan/2 Researcher/'
destdir = '/home/imeunu96/diffraction_patterns/Titan/'
count = 0

for folder, filename in walk(walkdir):
    filepath = os.path.join(folder, filename).replace("\\","/")
    paths = os.path.normpath(filepath).split(os.sep)
    # if paths[5] in prev_dirs:
    #   continue
    ext = os.path.splitext(filename)[-1]
    if not isdm3(ext):
        continue
    try:
        s = hs.load(escape_square_brackets(filepath))
        if not isdiff(s):
            continue
        paths = os.path.normpath(filepath).split(os.sep)
        newdir = os.path.join(destdir,'/'.join(paths[5:-1]))
        os.makedirs(newdir, exist_ok=True)
        if not os.path.isfile(os.path.join(newdir, filename)):
            shutil.copy2(filepath, newdir)
            count += 1
            print('%s' % os.path.join(newdir, filename))
        if count>=10000:
            break
    except Exception as e:
        print("Exception : %s in %s" % (str(e),filepath))
