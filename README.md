# Classification of TEM Diffraction Patterns via Deep Learning and Computer Vision

###### Sorry that I cannot upload data or images due to copyright and security reasons.

## Introduction
In Crystallography, Transmission Electron Microscopy(TEM) is a microscopy technique that allows us to analyze crystal structure by diffraction pattern.
<br>
The two largest categories of diffraction patterns are Ring Pattern and Spot Pattern. Each of them requires different analyze method. In order to automate the analysis, it was necessary to develop an A.I. model that can classify them.
I have tried Rule-based Method and Neural Network Models.

Diffraction Patterns example : https://myscope.training/legacy/tem/background/concepts/imagegeneration/diffractionimages.php

## [Rule based Method](https://github.com/imeunu/KIST/tree/main/RuleBased)
The way to intuitively distinguish them through [computer vision](https://github.com/imeunu/KIST/blob/main/RuleBased/computervision.py) is whether or not a circle is detected. The methods used to detect circle is Hough Transformation and RANSAC.
Also, when Fast Fourier Transform is performed, the shape of a circle appears as a line, and attempted to detect a straight line.
But found out that different images have different contrast. Thus it was necessary to adjust the contrast of the images collectively.
<br>
CLAHE was used to create reference data with well-contrast adjustment, and histogram matching is applied to the reference data for all other images.
After adjusting the contrast, [Hough Trasformation](https://github.com/imeunu/KIST/blob/main/RuleBased/total_investigation.py) performed the best of the three methods.
Accuracy was about 82.82%

## [Deep Learning Model](https://github.com/imeunu/KIST/tree/main/Deep%20Learning)
Used basic [CNN](https://github.com/imeunu/KIST/blob/main/Deep%20Learning/CNN.py) model and models known to be highly accurate in recent image analysis such as [GoogleNet](https://github.com/imeunu/KIST/blob/main/Deep%20Learning/GoogleNet.py), [DenseNet](https://github.com/imeunu/KIST/blob/main/Deep%20Learning/DenseNet.py).
The final result was derived by averaging the results of the three models using an [ensemble voting classifier](https://github.com/imeunu/KIST/blob/main/Validation/ensemble.py).
Accuracy was nearly 99.798%

## Conclusion
Deep learning model performed better than the rule-based method. With [Grad-CAM](https://github.com/imeunu/KIST/blob/main/Validation/Grad-CAM.py), we can see that the model predicted through center beam or scattered spots.
Through developed model, we can classify unknown TEM diffraction pattern as ring and spot pattern. I believe it can contribute to the automation of analysis of TEM.
