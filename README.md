# Classification of TEM Diffraction Patterns via Deep Learning and Computer Vision

## Introduction
In Crystallography, Transmission Electron Microscopy(TEM) is a microscopy technique that allows us to analyze crystal structure by diffraction pattern.
The two largest categories of diffraction patterns are Ring Pattern and Spot Pattern. Each of them requires different analyze method. In order to automate the analysis, it was necessary to develop an A.I. model that can classify them.
I have tried Rule-based Method and Neural Network Models.

## Rule based Method
The way to intuitively distinguish them is whether or not a circle is detected. The methods used to detect circle is Hough Transformation and RANSAC.
Also, when Fast Fourier Transform is performed, the shape of a circle appears as a line, and attempted to detect a straight line.

## Deep Learning Model


## Conclusion
