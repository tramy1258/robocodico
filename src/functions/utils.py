import numpy as np
import cv2
#import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.patches as patches

colors = ['orchid','tomato','olive','cadetblue','cornflowerblue','goldenrod','darkseagreen','crimson','lightpink']

#### Preprocessing ####
def black_white(image,t=None,otsu=False):
    if t is None:
        if otsu:
            t,_ = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            return np.where(image > t+30, 255, 0)
        else:
            return np.where(image > np.mean(image), 255, 0)
    else:
        return np.where(image > t+20, 255, 0)

# def phansalkar(image,n=5,p=3,q=10,k=0.25,R=0.5):
#     dx, dy = image.shape

    
def subsample(image,rate=2):
    return image[::rate]

def normalize(image):
    return image/255.0

def convolve2d(image,w):
    return cv2.filter2D(image, -1, w, borderType=cv2.BORDER_CONSTANT)

def sharpen(image):
    w = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(image, -1, w, borderType=cv2.BORDER_CONSTANT)

# perform a fast fourier transform and create a scaled, frequency transform image
def ft_image(norm_image):
    f = np.fft.fft2(norm_image) 
    fshift = np.fft.fftshift(f)
    frequency_tx = 20*np.log(np.abs(fshift))

    return frequency_tx

def show_mean_bi(image):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (12,3))
    ax1.imshow(image,cmap='gray')
    ax2.plot(np.mean(image,0))
    ax3.plot(np.mean(image,1))
    plt.show()

#### Evaluating ####
def show_splits(image,splits,ax=None,color='red',gray=False):
    """ Show image with splits given in input
    """
    if ax is None:
        plt.figure(figsize = (20,10))
        if gray:
            plt.imshow(image,cmap='gray')
        else:
            plt.imshow(image)
        plt.autoscale(False)
        for (axis,t) in splits:
            if axis == 'x':
                plt.axvline(t,linewidth=1,color=color)
            elif axis == 'y':
                plt.axhline(t,linewidth=1,color=color)   
        plt.show()  
    else:
        if gray:
            ax.imshow(image,cmap='gray')
        else:
            ax.imshow(image)
        ax.autoscale(False)
        for (axis,t) in splits:
            if axis == 'x':
                ax.axvline(t,linewidth=1,color=color)
            elif axis == 'y':
                ax.axhline(t,linewidth=1,color=color) 
        #plt.show()  

def show_areas(image,areas,ax=None,color='red',gray=False):
    if ax is None:
        plt.figure(figsize = (20,10))
        if gray:
            plt.imshow(image,cmap='gray')    
        else:
            plt.imshow(image)
        plt.autoscale(False)
        for x1,y1,x2,y2 in areas:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='red', fill=False)
            plt.gca().add_patch(rect)
        plt.show()
    else:
        if gray:
            ax.imshow(image,cmap='gray')
        else:
            ax.imshow(image)
        ax.autoscale(False)
        for x1,y1,x2,y2 in areas:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='red', fill=False)
            ax.add_patch(rect)

def show_linked_areas(image,linked_areas,ax=None,gray=False):
    if ax is None:
        plt.figure(figsize = (20,10))
        if gray:
            plt.imshow(image,cmap='gray')    
        else:
            plt.imshow(image)
        plt.autoscale(False)
        for i,a in linked_areas.items():
            for s in a[0]:
                x1,y1,x2,y2 = s[:4]
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=colors[i], fill=False)
                plt.gca().add_patch(rect)
        plt.show()
    else:
        if gray:
            ax.imshow(image,cmap='gray')
        else:
            ax.imshow(image)
        ax.autoscale(False)
        for i,a in linked_areas.items():
            for s in a[0]:
                x1,y1,x2,y2 = s[:4]
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=colors[i], fill=False)
                ax.add_patch(rect)

def splits_to_areas(splits):
    x = [int(s[1]) for s in splits if s[0] == 'x']
    x.sort()
    y = [int(s[1]) for s in splits if s[0] == 'y']
    y.sort()
    res = []
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            res.append([x[i],y[j],x[i+1],y[j+1]])
    return res


def eval_splits(image,splits):
    pass
