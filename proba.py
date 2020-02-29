# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:13:36 2019

@author: ASUS
"""

import pydicom
import numpy as np
import os
import matplotlib.pyplot as plt


# A)


PathDicom = os.getcwd()

pathHealthy = PathDicom + '\\zdrav'

healthy = []
i = 0

for dirName, subdirList, fileList in os.walk(pathHealthy):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            healthy.append(pydicom.dcmread(os.path.join(dirName,filename)))
            
prvi = healthy[0]

size_prvi = np.size(prvi.pixel_array)

# Get ref file
RefDs = healthy[0]

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), 16)

# Load spacing values (in mm)
#ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

# The array is sized based on 'ConstPixelDims'
ArrayDicom = np.zeros(ConstPixelDims)

# loop through all the DICOM files
ArrayDicom = prvi.pixel_array

plt.imshow(ArrayDicom[0,:,:], cmap = 'gray')