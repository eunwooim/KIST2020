# Classification of TEM Diffraction Patterns via Deep Learning and Computer Vision

## Introduction
In Crystallography, Transmission Electron Microscopy(TEM) is a microscopy technique that allows us to analyze crystal structure by diffraction pattern.
The two largest categories of diffraction patterns are Ring Pattern and Spot Pattern. Each of them requires different analyze method. In order to automate the analysis, it was necessary to develop an A.I. model that can classify them.
I have tried Rule-based Method and Neural Network Models.

## Rule based Method
The way to intuitively distinguish them is whether or not a circle is detected. The methods used to detect circle is Hough Transformation and RANSAC.
Also, when Fast Fourier Transform is performed, the shape of a circle appears as a line, and attempted to detect a straight line.
But found out that different images have different contrast. Thus it was necessary to adjust the contrast of the images collectively.
CLAHE was used to create reference data with well-contrast adjustment, and match histograms were applied to the reference data for all other images.
Accuracy was about 82.82%

## Deep Learning Model
Used basic CNN model and models known to be highly accurate in recent image analysis such as GoogleNet, DenseNet.
The final result was derived by averaging the results of the three models using an ensemble voting classifier.
Accuracy was nearly 99.798%

## Conclusion
