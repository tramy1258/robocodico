import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .utils import *
from .cart import ponderated_criterion
from sklearn.metrics import mean_squared_error
from scipy.ndimage import rotate

def prepare_image(img,dir,cvd):
    '''
    Parameters:
        - img : 2D array
        - dir : char in {'m','l','r','u','d'}
        - cvd : boolean
    Returns:
        Prepare image for splitting or trimming.
    '''
    if len(img.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        image = img.copy()

    # undo normalization
    if (image<=1).all():
        image = image*255
        image = image.astype(int)
        image = np.uint8(image)
    
    # convolve
    if cvd:
        image = sharpen(image)

    # transpose
    tr = False
    if (dir == 'm' and len(image) > len(image[0])) or dir in {'u','d'}: # only consider splitting along longer edge and trimming left or right
        image = cv2.transpose(image)
        tr = True

    return image,tr

def hough_vertical_lines(edges,angle=np.pi/9,threshold=200):
    '''
    Parameters:
        - band : 2D array
        - angle : float

    Returns:
        Normal vector and reference point for each vertical and almost vertical line found using Hough Transform.
    '''
    # find all lines
    lines = cv2.HoughLines(edges,1,np.pi/180,threshold)
    if lines is None:
        return None,None,None,None
    lines = lines.squeeze(axis=1)

    # filter lines out of given vertical angle limits
    a = np.cos(lines[:,1])
    b = np.sin(lines[:,1])
    filter = b < np.sin(angle)
    x0 = ((a*lines[:,0])[filter]).astype(int)
    y0 = ((b*lines[:,0])[filter]).astype(int)
    a = a[filter]
    b = b[filter]
    # print([l if l<np.pi/2 else np.pi-l for l in lines[filter][:,1]])
    # print(len(a), 'candidate lines amongst',len(lines),'lines found.')

    return a,b,x0,y0,lines[filter][:,1]

def get_points(n,m,a,b,x0,y0,angle=None):
    '''
    Parameters:


    Returns:
        Compute all points coordinates in a line
    '''
    x = []
    y = []
    if b == 0: # vertical line
        x = [int(x0) for _ in range(n)]
        y = [y for y in range(n)]

    else:
    # elif b < np.sin(angle): # almost vertical line
        x_ = [int((-b/a)*y + (b/a)*y0 + x0) for y in range(n)]
        x = [x for x in x_ if x < m and x >= 0]
        y = [y for y in range(n) if x_[y] < m and x_[y] >= 0]

    # else:
        # print(b - np.sin(angle)) #  show error
        # print(f'This angle ({a},{b},sin={np.sin(angle)}) is not fitting for this direction, this should not happen.')

    return x,y

# def hough_line_point(theta,rho,x=None,y=None):
#     '''
    
#     '''
#     if x is not None and y is not None:
#         return None
#     if y is not None:
#         return (-np.sin(theta)/np.cos(theta)*y + rho/np.cos(theta))
#     if x is not None:
#         return (-np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta))

# def filter_borders(img,lines,c=5,ax=None):
#     '''
#     Returns:
#         Filter all lines crossing middle of image
#     '''
#     print(lines.shape)
#     upper_y = img.shape[0] // c
#     lower_y = img.shape[0] // c * (c-1)
#     upper_x = img.shape[1] // c
#     lower_x = img.shape[1] // c * (c-1)
#     upper_xs = hough_line_point(lines[:,1],lines[:,0],y=np.array([upper_y]*len(lines)))
#     lower_xs = hough_line_point(lines[:,1],lines[:,0],y=np.array([lower_y]*len(lines)))
#     upper = np.logical_and(upper_xs > upper_x, upper_xs < lower_x)
#     lower = np.logical_and(lower_xs > upper_x, lower_xs < lower_x)
#     if ax is not None:
#         rect = patches.Rectangle((upper_x,upper_y),lower_x-upper_x,lower_y-upper_y,color='red')
#         ax.add_patch(rect)
#         ax.scatter(upper_xs, np.array([upper_y]*len(lines)),marker='x',color='powderblue')
#         ax.scatter(lower_xs, np.array([lower_y]*len(lines)),marker='x',color='plum')

#     return np.logical_or(upper, lower)

# def split(img,ratio=0.8,c=7,angle = np.pi/9,cvd=False,verbose=True):
#     '''
#     Parameters:

#     Returns:
#         Split if given image of 2 pages, else return original image.
#     '''

#     image,tr = prepare_image(img,'m',cvd)

#     if verbose:
#         fig, (ax,d,e,f) = plt.subplots(1,4,figsize=(16,5),width_ratios=[5,2,2,2])
#         ax.imshow(image,cmap='gray')

#     x1 = int((len(image[0])//c)*np.ceil(c/2-1)) 
#     y1 = 0
#     x2 = int(len(image[0])-(len(image[0])//c)*np.ceil(c/2-1))
#     y2 = len(image)

#     band = image[y1:y2,x1:x2]
#     bw = black_white(band,np.mean(band))
#     edges = cv2.Canny(band,30,150,apertureSize = 3)

#     if verbose:
#         ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', fill=False))
#         d.imshow(band,cmap='gray')
#         e.imshow(edges,cmap='gray')
#         f.imshow(bw,cmap='gray')
#         plt.autoscale(False)
    
#     a,b,x0,y0,l = hough_vertical_lines(band,angle)

#     # compute ratio of each line found to get one splitting position
#     split = -1
#     rmax = 0.0
#     xmax = []
#     ymax = []
#     for j in range(len(a)):
        
#         x,y = get_points(band,a[j],b[j],x0[j],y0[j],angle)

#         # filter lines not wholly in image
#         if len(y) == 0 or min(y) > 0 or max(y) < len(band)-1:
#             continue

#         # compute ratio
#         try:
#             r = 1-(np.sum([bw[y[i],x[i]] for i in range(len(y))]))/len(y)
#         except Exception as e:
#             print('EXCEPTION:',e)
#             print(a[j],b[j],x0[j],y0[j])
#             print(x,y)
#             if verbose:
#                 f.plot(x,y,color='powderblue')
#                 plt.show()
#             return

#         if verbose:
#             d.plot(x,y,color='tomato')
#         #rs.append(r)
#         if rmax < r:
#             rmax = r
#             xmax = x
#             ymax = y

#         # compute splitting position
#         if r >= ratio:
#             split = x1 + x[len(x)//2]
#             print('r =',r, 'with split at',split)
#             if verbose:
#                 f.axvline(x[len(x)//2],color='red',linestyle='dashed')
#                 f.plot(x,y,color='tomato')
#                 plt.show()
#             break

#     print('rmax =',rmax,'r_=',(np.sum([edges[ymax[i],xmax[i]] for i in range(len(ymax))]))/len(ymax))
#     if verbose:
#         f.plot(xmax,ymax,color='darkcyan')
#         plt.show()

#     if split != -1:
#         if not tr:
#             img1 = img[:,:split]
#             img2 = img[:,split:]
#         else:
#             img1 = img[:split]
#             img2 = img[split:]
#         res = [img1,img2]
#         if verbose:
#             fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
#             ax1.imshow(img1,cmap='gray')
#             ax2.imshow(img2,cmap='gray')
#             plt.show()
#     else:
#         res = [img]
#         if verbose:
#             plt.imshow(img,cmap='gray')
#             plt.show()

#     return res

# def trim(img,dir,ratio=0.7,c=6,angle=np.pi/9,cvd=False,verbose=True):
#     ''' 
#     Parameters:
#         - img: 2D array
#         - ratio: float
#         - c: int
#         - cvd: boolean
#         - verbose: boolean
#         - dir: char
    
#     Returns:
#         Trim image in given direction left (l), right (r), up (u), down (d).
#         Returns 2 images and cropping position if cropping is reasonable, one image and -1 otherwise.
#     '''

#     print(f'---------{dir}----------')
#     image,tr = prepare_image(img,dir,cvd)

#     if verbose:
#         fig, (ax,d,e,f) = plt.subplots(1,4,figsize=(16,5),width_ratios=[5,2,2,2])
#         ax.imshow(image,cmap='gray')
#         ax.autoscale(False)

#     x1 = 0
#     y1 = 0
#     x2 = len(image[0])
#     y2 = len(image)
#     if dir == 'l' or dir == 'u': # left vertival line
#         x2 = len(image[0])//c
#     elif dir == 'r' or dir == 'd': # right vertical line
#         x1 = (len(image[0])//c)*(c-1)
#     else:
#         print('No such direction exists.')
#         return

#     # showing region to consider cutting
#     band = image[y1:y2,x1:x2]
#     bw = black_white(band,np.mean(band))
#     edges = cv2.Canny(band,30,150,apertureSize = 3)
#     if verbose:
#         ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', fill=False))
#         d.imshow(band,cmap='gray')
#         d.autoscale(False)
#         f.imshow(bw,cmap='gray')

#     if verbose:
#         a,b,x0,y0,_ = hough_vertical_lines(band,angle,e)
#     else:
#         a,b,x0,y0,_ = hough_vertical_lines(band,angle)
#     split = -1
#     if a is not None:
#         # compute ratio of each line found to see if a rupture exists
#         rmax = 0.0
#         xmax = []
#         ymax = []
#         rbest = 0.0
#         xbest = []
#         ybest = []
#         riskbest = -1
#         bbest = -1
#         #rs = []

#         for j in range(len(a)):

#             x,y = get_points(band,a[j],b[j],x0[j],y0[j],angle)

#             # only consider lines wholly in image
#             if len(y) == 0 or min(y) > 0 or max(y) < len(band)-1:
#                 continue

#             # compute ratio
#             try:
#                 r = 1-(np.sum([bw[y[i],x[i]] for i in range(len(y))]))/len(y)
#             except Exception as e:
#                 print('EXCEPTION:',e)
#                 print(a[j],b[j],x0[j],y0[j])
#                 print(x,y)
#                 if verbose:
#                     f.plot(x,y,color='powderblue')
#                     plt.show()
#                 return
                
#             if verbose:
#                 d.plot(x,y,color='tomato')
#                 #f.plot(x,y,color='tomato')

#             if rmax < r:
#                 rmax = r
#                 xmax = x
#                 ymax = y
#                 bbest = b[j]

#             # compute splitting position
#             if r >= ratio:
#                 risk = ponderated_criterion(band,mean_squared_error,x[len(x)//2],'x')
#                 #print('r =',r, 'with split at',x1 + x[len(x)//2],'with risk =',risk)
#                 if riskbest == -1 or risk < riskbest:
#                     riskbest = risk
#                     rbest = r
#                     xbest = x
#                     ybest = y
#                     split = x1 + x[len(x)//2]
#                 if verbose:
#                     f.axvline(x[len(x)//2],color='darkcyan',linestyle='dashed')
#                 #print('r =',r, 'with split at',x1 + x[len(x)//2],'with risk =',risk)
        
#         print('rmax =',rmax,'r_=',(np.sum([edges[ymax[i],xmax[i]] for i in range(len(ymax))]))/len(ymax))
#         print('r =',rbest, 'with split at',split,'with risk =',riskbest,'and sin(theta)=',bbest)
#         if verbose:
#             if len(xbest) > 0:
#                 f.plot(xbest,ybest,color='tomato')
#                 f.axvline(xbest[len(xbest)//2],color='red',linestyle='dashed')
    
#     if verbose:    
#         plt.show()
#     if split == -1:
#         if dir == 'l' or dir == 'u':
#             return 0
#         elif dir == 'r':
#             return len(img[0])
#         elif dir == 'd':
#             return len(img)

#     return split

# def reframe(image,split_only=False,verbose=True):
#     '''
#     '''
#     imgs = split(image,verbose=verbose,ratio=0.75)
#     if not split_only:
#         for i in range(len(imgs)):
#             c = []
#             for d in ['l','r','u','d']:
#                 pos = trim(imgs[i],d,verbose=verbose)
#                 print(pos)
#                 c.append(pos)
#             if len(imgs[i]) > len(imgs[i][0]):
#                 imgs[i] = imgs[i][c[2]:c[3],c[0]:c[1]]
#             else:
#                 imgs[i] = imgs[i][c[0]:c[1],c[2]:c[3]]
#             print(c)
#     fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(14,7))
#     ax1.imshow(image,cmap='gray')
#     ax = [ax2,ax3]
#     for i in range(len(imgs)):
#         ax[i].imshow(imgs[i],cmap='gray')
#     plt.show()
#     return imgs

def get_band_limits(image,dir,c):
    '''
    '''
    x1 = 0
    y1 = 0
    x2 = len(image[0])
    y2 = len(image)
    if dir == 'l' or dir == 'u': # left vertival line
        x2 = len(image[0])//c
    elif dir == 'r' or dir == 'd': # right vertical line
        x1 = (len(image[0])//c)*(c-1)
    elif dir == 'm':
        x1 = int((len(image[0])//c)*np.ceil(c/2-1))
        x2 = int(len(image[0])-(len(image[0])//c)*np.ceil(c/2-1))
    else:
        print('No such direction exists.')
        return 
    return x1, y1, x2, y2

def get_good_lines(band,bw,x1,ratio,a,b,x0,y0,theta,angle=None,d=None,e=None,f=None):
    '''
    '''
    if (bw>1).any():
        bw = normalize(bw).astype(int)

    all_rbw = np.array([])
    all_x = np.array([])
    all_y = np.array([]) 
    all_risk = np.array([])
    all_angle = np.array([])

    for j in range(len(a)):
        for delta in [0,5,-5]:

            if 0 > x0[j] + delta or x0[j] + delta >= len(band[0]):
                continue

            x,y = get_points(len(band),len(band[0]),a[j],b[j],x0[j]+delta,y0[j]) #,angle)

            # filter lines not entirely in image
            if len(y) == 0 or min(y) > 0 or max(y) < len(bw)-1:
                continue

            if e is not None:
                e.plot(x,y,color='tomato')

            # compute ratio
            try:
                rbw  = 1-(np.sum([bw[y[i],x[i]] for i in range(len(y))]))/len(y)
                #rbwf = 1-(np.sum([bw[y[i],x[i]-5] for i in range(len(y))]))/len(y)
                #rbwb = 1-(np.sum([bw[y[i],x[i]+5] for i in range(len(y))]))/len(y)
                
            except Exception as e:
                print('EXCEPTION:',e)
                print(a[j],b[j],x0[j],y0[j])
                print(len(x),len(y))
                print(x,y)
                if e is not None:
                    print('printpowderblue')
                    e.plot(x,y,color='powderblue')
                    plt.show()
                return
            # print(rbw,rbwf,rbwb)
            if rbw >= ratio:# or rbwf >= ratio or rbwb >= ratio:
                all_rbw = np.append(all_rbw,rbw)
                all_x = np.append(all_x,x1 + x[len(x)//2]-delta)
                all_y = np.append(all_y,y[len(y)//2])
                all_risk = np.append(all_risk,ponderated_criterion(band,mean_squared_error,x[len(x)//2],'x'))
                all_angle = np.append(all_angle,theta[j])
                if f is not None:
                    f.plot(x,y,color='teal')
                continue
    # all_angle = np.where(all_angle > np.pi/2, np.pi - all_angle, all_angle)
    return all_rbw, all_x, all_y, all_risk, all_angle


def get_split(img,dir,ratio=0.7,c=6,angle=np.pi/9,verbose=True,cvd=False):
    '''
    '''
    image, tr = prepare_image(img,dir,cvd)
    x1, y1, x2, y2 = get_band_limits(image,dir,c)

    band = image[y1:y2,x1:x2]
    bw = black_white(band) #,otsu=True)
    edges = cv2.Canny(band,30,150,apertureSize = 3)    
    d = e = f = None

    if verbose:
        _, (ax,e,f,d) = plt.subplots(1,4,figsize=(16,5),width_ratios=[5,2,2,2])
        ax.imshow(image,cmap='gray')
        ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', fill=False))
        d.imshow(band,cmap='gray')
        e.imshow(edges,cmap='gray')
        f.imshow(bw,cmap='gray')
        ax.autoscale(False)
        plt.autoscale(False)

    a, b, x0, y0, theta = hough_vertical_lines(edges,angle)
    all_rbw, all_x, all_y, all_risk, all_angle = get_good_lines(band,bw,x1,ratio,a,b,x0,y0,theta,d=d,e=e,f=f)   

    if len(all_rbw) < 1 and verbose:
        plt.show()

    if dir == 'm':
        if len(all_rbw) < 1:
            return None, None , None
        i = np.argmax(all_rbw)
    else:
        if len(all_rbw) < 1:
            if dir in {'u','l'}:
                return 0, (y2 - y1)/2, 0
            else:
                return x2, (y2 - y1)/2, 0
        i = np.argmin(all_risk) # get index of least risky split (splitted zone are more homogeneous)
    
    print(f'split at x={all_x[i]} and angle={round(all_angle[i],5)} with black-white ratio {round(all_rbw[i],5)}')

    if verbose:
        x_ = all_x[i]
        y_ = all_y[i]
        a_ = -all_angle[i] - np.pi/2
        slope_ = -np.sin(a_)/np.cos(a_)
        ax.axline((x_,y_),slope=slope_,linestyle='dashed')
        ax.scatter(x_,y_,marker='x',color='red')
        d.axline((x_-x1,y_),slope=slope_,linestyle='dashed')
        d.scatter(x_-x1,y_,marker='x',color='red')
        plt.show()
        
    return all_x[i], all_y[i], all_angle[i]

def all_orientation(img,ratio=0.7,c=4,angle=np.pi/9,verbose=True,cvd=False):
    '''
    '''
    x = np.array([])
    y = np.array([])
    a = np.array([])
    for dir in ['l','r','u','d']:
        x_,y_,a_ = get_split(img,dir,ratio,c,angle,verbose,cvd)
        x = np.append(x,x_) 
        y = np.append(y,y_)
        if dir in {'l', 'r'}:
            a = np.append(a,(np.pi-a_)%np.pi)
        else:
            a = np.append(a,a_)
    return x,y,a

def main_orientation(angle,atol=2e-2):
    '''
    Parameters: angles of 4 sides
        
    '''
    sim = dict()
    for i in range(len(angle)):
        print(sim)
        # if angle[i] is None:
            # continue
        have_sim = False
        for j in sim:
            # print('i',i,'j',j,end='  ')
            for a in sim[j]:
                # print(np.abs(a-angle[i]),np.abs(-a+angle[i]-np.pi))
                if np.abs(a-angle[i]) < atol:
                    have_sim = True
                    sim[j].append(angle[i])
                elif np.abs(-a+angle[i]-np.pi) < atol:
                    have_sim = True
                    sim[j].append(-np.pi+angle[i])
                if have_sim:
                    break
        if not have_sim:
            sim[i] = [angle[i]]
    print(sim)
    for j in sim:
        if len(sim[j]) > 2:
            return np.mean(sim[j])
        
    return 0 # if no main orientation can be found, do not rectify at all

def rotate_and_reframe(img,rad,x,y,verbose=True):
    '''
    '''
    height, width = img.shape[:2]
    center = (width/2,height/2)
    rad = rad - np.pi if rad > np.pi/2 else rad
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-rad*180/np.pi, scale=1)

    abs_cos = abs(rotate_matrix[0,0]) 
    abs_sin = abs(rotate_matrix[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    # rotate_matrix[0, 2] += bound_w/2 - center[0]
    # rotate_matrix[1, 2] += bound_h/2 - center[1]
    rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(bound_w,bound_h))

    x_1 = (x*rotate_matrix[0,0] + y*rotate_matrix[0,1] + rotate_matrix[0,2]).astype(int)
    y_1 = (x*rotate_matrix[1,0] + y*rotate_matrix[1,1] + rotate_matrix[1,2]).astype(int)
      
    x_2 = (y*rotate_matrix[0,0] + x*rotate_matrix[0,1] + rotate_matrix[0,2]).astype(int)
    y_2 = (y*rotate_matrix[1,0] + x*rotate_matrix[1,1] + rotate_matrix[1,2]).astype(int)

    x_ = np.concatenate((x_1[:2],x_2[2:]))
    y_ = np.concatenate((y_1[:2],y_2[2:]))
        
    if verbose:
        fig, (a,b) = plt.subplots(1,2,figsize=(20,10))

        a.imshow(img)
        a.scatter(x[:2],y[:2],marker='x',color='red')
        a.scatter(y[2:],x[2:],marker='x',color='red')
        a.autoscale(False)

        b.imshow(rotated_image)
        b.axvline(x_[0],color='tomato',linestyle='dashed')
        b.axvline(x_[1],color='tomato',linestyle='dashed')
        b.axhline(y_[2],color='tomato',linestyle='dashed')
        b.axhline(y_[3],color='tomato',linestyle='dashed')
        b.scatter(x_,y_,marker='x',color='red')
        b.autoscale(False)
        plt.show()

    return x_[0],x_[1],y_[2],y_[3]
    
def rectify(img):
    '''
    '''
    pass

def highlight_fond_color(image,color='tomato'):
    hist,vals = np.histogram(image,bins=50)
    print('hist =',hist,np.sum(hist),len(hist))
    print('vals =',vals,len(vals))
    i = np.argmax(hist)
    print(vals[i],vals[i+1])
    xy = [(x,y) for y in range(len(image)) for x in range(len(image[0])) if (image[y,x]>=vals[i] and image[y,x]<vals[i+1])]
    xy = np.array(xy)
    plt.figure(figsize=(20,10))
    plt.imshow(image,cmap='gray')
    plt.scatter(xy[:,0],xy[:,1],color=color,marker='.',s=1)
    plt.show()