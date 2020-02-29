# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:40:21 2019

@author: Ivana Munjas
"""

import pydicom
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import SimpleITK as sitk


# =============================================================================
# a) ucitavanje fajlova
# =============================================================================


# Kod treba sacuvati u folderu gde se nalaze i folderi za zdrave i transplatirane
# bubrege. Sledeca komanda dohvata putanju gde se nalazi .py file
PathDicom = os.getcwd()

# Ucitavanje MRI slika za zdrave ispitanike

pathHealthy = PathDicom + '\\zdrav'

K_healthy = []
T_healthy = []
PDW_healthy = []

i = 0

for dirName, subdirList, fileList in os.walk(pathHealthy):
    for filename in fileList:
        if ".dcm" in filename.lower():
            if i<12 and i%2 == 0:
                K_healthy.append(pydicom.dcmread(os.path.join(dirName,filename)))
            elif i<12 and i%2 != 0:
                T_healthy.append(pydicom.dcmread(os.path.join(dirName,filename)))
            else:
                PDW_healthy.append(pydicom.dcmread(os.path.join(dirName,filename)))
            i = i + 1
        
# Ucitavanje MRI slika za ispitanike sa transplatiranim bubregom

pathTransplanted = PathDicom + '\\transplantiran'

K_transplanted = []
T_transplanted = []
PDW_transplanted = []

i = 0

for dirName, subdirList, fileList in os.walk(pathTransplanted):
    for filename in fileList:
        if ".dcm" in filename.lower():
            if i<12 and i%2 == 0:
                K_transplanted.append(pydicom.dcmread(os.path.join(dirName,filename)))
            elif i<12 and i%2 != 0:
                T_transplanted.append(pydicom.dcmread(os.path.join(dirName,filename)))
            else:
                PDW_transplanted.append(pydicom.dcmread(os.path.join(dirName,filename)))
            i = i + 1
            
# =============================================================================
# b) odredjivanje srednje vrednosti po slajsevima
# =============================================================================
            
ConstPixelDims = (int(K_healthy[0].Rows), int(K_healthy[0].Columns), 16)

Kmean_h = np.zeros(ConstPixelDims)
Tmean_h = np.zeros(ConstPixelDims)

Kmean_t = np.zeros(ConstPixelDims)
Tmean_t = np.zeros(ConstPixelDims)

f_h = np.zeros(ConstPixelDims)
f_t = np.zeros(ConstPixelDims)

for i in range(0,6):
    k_h_array = K_healthy[i].pixel_array
    t_h_array = T_healthy[i].pixel_array
    k_t_array = K_transplanted[i].pixel_array
    t_t_array = T_transplanted[i].pixel_array
    for j in range(0,16):
        Kmean_h[:,:,j] += k_h_array[j,:,:]
        Tmean_h[:,:,j] += t_h_array[j,:,:]
        Kmean_t[:,:,j] += k_t_array[j,:,:]
        Tmean_t[:,:,j] += t_t_array[j,:,:]
        
Kmean_h = Kmean_h/6
Tmean_h = Tmean_h/6
Kmean_t = Kmean_t/6
Tmean_t = Tmean_t/6


# =============================================================================
# c) racunanje parametarskih slika 
# =============================================================================
l = 80/100
alfa = 1
TI_h = 1200
TI_t = 1990
T1 = 1150

fig = plt.figure(figsize = (8,8))
plt.title('Parametarske slike preklopljene sa PDW slikama kod zdravog ispitanika')
plt.axis('off')

# S obzirom da su neke vrednosti PDW slika jednake nuli, vrsila sam proveru za 
# svaki piksel da li je jednak nuli ili ne. Ako jeste onda se deljenje vrsi sa 0.0001
# umesto nedozvoljenog deljenja nulom.
for i in range(0,16):
    for j in range(0,128):
        for k in range(0,128):
            if (PDW_healthy[i].pixel_array)[j,k] != 0:
                f_h[j,k,i] = 60000*l/(2*alfa*TI_h*0.001)*(Kmean_h[j,k,i] - Tmean_h[j,k,i])/(PDW_healthy[i].pixel_array)[j,k]*np.exp(TI_h/T1)
            else:
                f_h[j,k,i] = 60000*l/(2*alfa*TI_h*0.001)*(Kmean_h[j,k,i] - Tmean_h[j,k,i])/0.0001*np.exp(TI_h/T1)
            
            if f_h[j,k,i] < 0:
                f_h[j,k,i] = 0
                
                
            if (PDW_transplanted[i].pixel_array)[j,k] != 0:
                f_t[j,k,i] = 60000*l/(2*alfa*TI_t*0.001)*(Kmean_t[j,k,i] - Tmean_t[j,k,i])/(PDW_transplanted[i].pixel_array)[j,k]*np.exp(TI_t/T1)
            else:
                f_t[j,k,i] = 60000*l/(2*alfa*TI_t*0.001)*(Kmean_t[j,k,i] - Tmean_t[j,k,i])/0.0001*np.exp(TI_t/T1)
            
            if f_t[j,k,i] < 0:
                f_t[j,k,i] = 0
            
    # promena kontrasta parametarskih slika            
    im_mask = np.where(f_h[:,:,i].flatten() > 90000,0,1) 
    im_mask = im_mask.reshape(128,128)
    f_h[:,:,i] = f_h[:,:,i]*im_mask
    f_h[:,:,i] = f_h[:,:,i]/f_h[:,:,i].max()*255
    
    # prikaz parametarskih slika preklopljenih sa PDW slikama
    fig.add_subplot(4, 4, i+1)
    plt.imshow(PDW_healthy[i].pixel_array/PDW_healthy[i].pixel_array.max()*255, cmap = 'gray') 
    plt.imshow(f_h[:,:,i], cmap = 'jet', alpha = 0.6)
    plt.axis('off')

fig = plt.figure(figsize = (8,8))
plt.title('Parametarske slike preklopljene sa PDW slikama kod ispitanika sa transplatiranim bubregom')
plt.axis('off')

for i in range(0,16):
    im_mask = np.where(f_t[:,:,i].flatten() > 100000,0,1) 
    im_mask = im_mask.reshape(128,128)
    f_t[:,:,i] = f_t[:,:,i]*im_mask
    f_t[:,:,i] = f_t[:,:,i]/f_t[:,:,i].max()*255
    
    fig.add_subplot(4, 4, i+1)
    plt.imshow(PDW_transplanted[i].pixel_array/PDW_transplanted[i].pixel_array.max()*255, cmap = 'gray') 
    plt.imshow(f_t[:,:,i], cmap = 'jet', alpha = 0.6)
    plt.axis('off')
    
# =============================================================================
# d) resamplovanje parametarskih slika
# =============================================================================

resampled = []
fig = plt.figure(figsize = (8,8))
plt.title('Parametarske slike nakon resamplovanja')
plt.axis('off')

# Pomocu funkcije zoom mozemo specificirati kako zelimo da se vrsi resamplovanje
# u ovom slucaju sam odabrala da se ista vrednost piksela samo kopira na 8 piksela
# u novonastaloj slici 1024*1024. 

for i in range(0,16):
    resampled.append(ndi.zoom(f_h[:,:,i], 8, order=0))
    fig.add_subplot(4, 4, i+1)
    plt.imshow(resampled[i], cmap = 'jet')
    plt.axis('off')
    

# =============================================================================
# e) segmentacija bubrega
# =============================================================================
    
im = PDW_healthy[6].pixel_array

im = sitk.GetImageFromArray(im)
imgSmooth = sitk.CurvatureFlow(image1=im, timeStep=0.125, numberOfIterations=5)

# Pomocu biblioteke SimpleITK mogu se postaviti "semena" gde ce se raditi segmentacija
# na bazi rasta regiona tako sto ce se semenu dodavati pikseli koji spadaju u zadatu 
# granicu (lower,upper). Tako se automatski pravi maska koja ce se kasnije koristiti
# za racunanje renalne perfuzije

lstSeeds = [(42,80), (95,80)]
labelWhiteMatter = 1
imgWhiteMatter = sitk.ConnectedThreshold(image1=imgSmooth, seedList=lstSeeds, lower=400, upper=650, replaceValue=labelWhiteMatter)

# Postavila sam jos jedno seme da bi se otklonili nezeljeni delovi maske za desni bubreg

lstSeeds = [(113,60)]
imgWhiteMatter2 = sitk.ConnectedThreshold(image1=imgSmooth, seedList=lstSeeds, lower=450, upper=750, replaceValue=labelWhiteMatter)

imgWhiteMatter = imgWhiteMatter - imgWhiteMatter2
imgWhiteMatter = sitk.GetArrayFromImage(imgWhiteMatter)

im_open = ndi.binary_opening(imgWhiteMatter, iterations = 2)
im_dil = ndi.binary_dilation(ndi.binary_closing(im_open, iterations = 4), iterations = 1)

labeled_array, num_features = ndi.label(im_dil)


im = sitk.GetArrayFromImage(im)

plt.figure()
plt.imshow(im, cmap = 'gray') 
plt.imshow(labeled_array, alpha=0.15, cmap = 'magma') 
plt.title('Bubrezi nakon segmentacije')
plt.show()

kidney1 = np.where(labeled_array == 2, 1, 0)
kidney2 = np.where(labeled_array == 1, 1, 0)

RPl = np.sum(im*kidney1)/np.count_nonzero(kidney1)
RPr = np.sum(im*kidney2)/np.count_nonzero(kidney2)
print('Vrednost renalne perfuzije za levi bubreg: ' + str(RPl))
print('Vrednost renalne perfuzije za desni bubreg: ' + str(RPr))