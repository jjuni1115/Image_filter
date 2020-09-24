from PIL import Image
import numpy as np
from matplotlib import pyplot as plot
import random
import math
import statistics

SNR=9.0

def variance(input_image,n):     #variance function
    MN=len(input_image)*len(input_image[0])
    sum_x=0.0
    sum_x2=0.0
    for i in range(len(input_image)):
        for k in input_image[i]:
            sum_x2+=k**2
            sum_x+=k
    sigma_2=(sum_x2/float(MN))-((sum_x/float(MN))**2)
    return sigma_2

def AddGaussianNoise (input_img, noise_img, sigma):  #Add noise function
    for i in range(0,512):
        for j in range(0,512):
            s=input_img[i][j]+Gaussian(sigma)
            if s > 255:
                s = 255
            elif s < 0:
                s = 0
            noise_img[i][j]=s
def Gaussian(sd):     #Gaussian noise function
    ready=0
    gstore=0.0
    v1=0.0
    v2=0.0
    r=0.0
    fac=0.0
    gaus=0.0
    r1=0
    r2=0
    flag=0
    if ready==0:
        while 1:
            if flag==0:
                r1=random.randint(0,32767)
                r2=random.randint(0,32767)
                v1=2*(float(r1)/float(32767)-0.5)
                v2=2*(float(r2)/float(32767)-0.5)
                r=v1*v1+v2*v2
                flag=1
            elif flag==1 and r>1.0:
                r1 = random.randint(0, 32767)
                r2 = random.randint(0, 32767)
                v1 = 2 * (float(r1)/float(32767)-0.5)
                v2 = 2 * (float(r2)/float(32767)-0.5)
                r = v1 * v1 + v2 * v2
            else:
                break
        fac=float(math.sqrt(float(-2*math.log(r)/r)))
        gstore=v1*fac
        gaus=v2*fac
        ready=1
    else:
        ready=0
        gaus=gstore

    return gaus*sd

def median_filter(pixel):        #median filter function
    median=[]
    new_pixel = np.ones((512, 512), dtype=int)
    for i in range(0,512):
        for j in range(0,512):
            if i==0:
                new_pixel[i][j]=pixel[i][j]
            elif j==0:
                new_pixel[i][j] = pixel[i][j]
            elif i==511:
                new_pixel[i][j] = pixel[i][j]
            elif j==511:
                new_pixel[i][j] = pixel[i][j]
            else:
                for k in range(0,3):
                    for m in range(0,3):
                        median.append(pixel[i+k-1][j+m-1])
                #median.sort()
                new_pixel[i][j]=statistics.median(median)
                median.clear()
    return new_pixel

def low_pass_filter(pixel):             #low pass filter function
    new_pixel = np.ones((512, 512), dtype=int)
    for i in range(0,512):
        for j in range(0,512):
            if i==0:
                new_pixel[i][j]=pixel[i][j]
            elif j==0:
                new_pixel[i][j] = pixel[i][j]
            elif i==511:
                new_pixel[i][j] = pixel[i][j]
            elif j==511:
                new_pixel[i][j] = pixel[i][j]
            else:
                for k in range(0,3):
                    for m in range(0,3):
                        new_pixel[i][j]+=pixel[i+k-1][j+m-1]/9
    return new_pixel

def mean_square_error(pixel,newpixel):    #mean square error function
    sum=0
    mse=0.0
    for i in range(0,512):
        for j in range(0,512):
            sum+=(newpixel[i][j]-pixel[i][j])**2
    mse=sum/(512*512)
    return mse

mse=0.0
#load raw file
infile=open("BOAT512.raw","rb")
image=np.fromfile(infile,dtype=np.uint8,count=512*512)
pixel=Image.frombuffer('L',[512,512],image,'raw','L',0,1)
pixel=np.array(pixel)

#setting figure
fig=plot.figure(figsize=(10,7))
rows=1
cols=2

#Add Gaussian noise on image
noise_pixel=np.ones((512, 512), dtype=int)
var=variance(pixel,SNR)
stddev_noise=float(math.sqrt(var/math.pow(10.0,SNR/10)))
AddGaussianNoise(pixel,noise_pixel,stddev_noise)

#median filter
new_pixel = np.ones((512, 512), dtype=int)
new_pixel=median_filter(noise_pixel) #median filter
mse=mean_square_error(pixel,new_pixel) #calculate mse
print('median filter Mean Square Error:',mse)

#low_pass filter
new_pixel2=np.ones((512, 512), dtype=int)
new_pixel2=low_pass_filter(noise_pixel)  #low pass filter
mse=mean_square_error(pixel,new_pixel2)  #calculate mse
print('low-pass filter Mean Square Error:',mse)

#print image
im1=fig.add_subplot(rows,cols,1)  #original image
im1.imshow(pixel,cmap='gray')
im1.set_title('Original Image')
im1.axis('off')

im2=fig.add_subplot(rows,cols,2)   #noise image
im2.imshow(noise_pixel,cmap='gray')
im2.set_title('Noise Image')
im2.axis('off')
plot.show()

fig=plot.figure(figsize=(10,7))
rows=1
cols=2

im1=fig.add_subplot(rows,cols,1)  #median filter image
im1.imshow(new_pixel,cmap='gray')
im1.set_title('Median Filter')
im1.axis('off')

im2=fig.add_subplot(rows,cols,2)  #lowpass filter image
im2.imshow(new_pixel2,cmap='gray')
im2.set_title('Low-Pass filter')
im2.axis('off')

plot.show()