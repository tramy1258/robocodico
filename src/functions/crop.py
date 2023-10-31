import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .utils import *
from .cart import ponderated_criterion
from sklearn.metrics import mean_squared_error
from functions.analyse import count_up_down
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

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

    # Undo normalization
    if (image<=1).all():
        image = image*255
        # image = image.astype(int)
        image = np.uint8(image)
    
    # Convolve if necessary
    if cvd:
        image = sharpen(image)

    # transpose
    # tr = False
    # if (dir == 'm' and len(image) > len(image[0])) or dir in {'u','d'}: # only consider splitting along longer edge and trimming left or right
    if dir in {'u','d'}:
        image = cv2.transpose(image)
        # tr = True

    # return image,tr
    return image

def hough_vertical_lines(edges,angle=np.pi/9,nbmax=150):
    '''
    Parameters:
        - band : 2D array
        - angle : float

    Returns:
        Normal vector and reference point for each vertical and almost vertical line found using Hough Transform.
    '''
    # find all lines
    lines = cv2.HoughLines(edges,1,np.pi/180,int(len(edges)/7))
    if lines is None:
        return np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    lines = lines.squeeze(axis=1)

    # filter lines out of given vertical angle limits
    a = np.cos(lines[:,1])
    b = np.sin(lines[:,1])
    filter = b < np.sin(angle)
    x0 = ((a*lines[:,0])[filter]).astype(int)[:nbmax]
    y0 = ((b*lines[:,0])[filter]).astype(int)[:nbmax]
    a = a[filter][:nbmax]
    b = b[filter][:nbmax]
    # print([l if l<np.pi/2 else np.pi-l for l in lines[filter][:,1]])
    print(len(a), 'candidate lines amongst',len(lines),'lines found.')

    return a,b,x0,y0,lines[filter][:nbmax][:,1]

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

    return np.array(x),np.array(y)

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

def smooth_line(binary,x,y,window=5):
    '''
    '''
    line = [binary[y[i],x[i]] for i in range(len(y))]
    line = [1 if np.mean(line[i:i+window])>0.5 else 0 for i in range(0,len(line)-window)]
    return line

def get_good_lines(band,binary,x1,ratio,continuous_ratio,a,b,x0,y0,theta,d=None,e=None,f=None):
    '''
    '''
    if (binary>1).any():
        binary = normalize(binary).astype(int)

    all_rbw = np.array([])
    all_cont = np.array([])
    all_x = np.array([]).astype(int)
    all_y = np.array([]).astype(int)
    all_risk = np.array([])
    all_angle = np.array([])

    for j in range(len(a)):
        for delta in [0,5,-5]:

            if 0 > x0[j] + delta or x0[j] + delta >= len(band[0]):
            # if 0 > x0[j] or x0[j] >= len(band[0]):
                continue

            x,y = get_points(len(band),len(band[0]),a[j],b[j],x0[j]+delta,y0[j])
            # x,y = get_points(len(band),len(band[0]),a[j],b[j],x0[j],y0[j])

            # filter lines not entirely in image
            if len(y) == 0 or min(y) > 0 or max(y) < len(binary)-1:
                continue
            
            if e is not None:
                e.plot(x-delta,y,color='tomato')

            # compute ratio
            try:
                new_rbw = np.sum(smooth_line(binary,x,y))/len(y)
                rbw = np.sum([binary[y[i],x[i]] for i in range(len(y))])/len(y)
                cont_rbw_ = count_up_down([binary[y[i],x[i]] for i in range(len(y))])+[0]
                cont_rbw__ = cont_rbw_[1::2] if binary[y[0],x[0]] == 0 else cont_rbw_[::2]
                cont_rbw = np.max(np.array(cont_rbw__))/len(y)
                
            except Exception as err:
                print('EXCEPTION:',err)
                print(a[j],b[j],x0[j],y0[j])
                print(len(x),len(y))
                print(x,y)
                if e is not None:
                    e.plot(x-delta,y,color='powderblue')
                    plt.show()
                return
            # print('--'*50)
            # print('ratio =',rbw,'cont ratio =',cont_rbw,cont_rbw_)
            # print('ratio =',rbw,'smooth_ratio =',new_rbw, 'continuous_ratio =',cont_rbw, cont_rbw_,cont_rbw__)
            if rbw >= ratio:
                if f is not None:
                    f.plot(x-delta,y,color='tomato')
                if cont_rbw >= continuous_ratio:
                    # print('--->',rbw,cont_rbw,delta)
                    all_rbw = np.append(all_rbw,rbw)
                    all_cont = np.append(all_cont,cont_rbw)
                    all_x = np.append(all_x,x1 + x[len(x)//2]-delta)
                    # all_x = np.append(all_x,x1 + x[len(x)//2])
                    all_y = np.append(all_y,y[len(y)//2])
                    # all_risk = np.append(all_risk,ponderated_criterion(band,mean_squared_error,x[len(x)//2]-delta,'x'))
                    all_risk = np.append(all_risk,ponderated_criterion(band,mean_squared_error,x-delta))
                    all_angle = np.append(all_angle,theta[j])
                    if f is not None:
                        f.plot(x-delta,y,color='teal')
                        # print('---->',x1 + x[len(x)//2]-delta,y[len(y)//2],theta[j],rbw,cont_rbw)
                    break
    return all_rbw, all_x, all_y, all_risk, all_angle, all_cont


def get_split(img,dir,ratio=0.6,continuous_ratio=0.25,c=6,angle=np.pi/9,verbose=True,cvd=False,bnw=True,nbmax=100):
    '''
    '''
    # image, tr = prepare_image(img,dir,cvd)
    image = prepare_image(img,dir,cvd)
    x1, y1, x2, y2 = get_band_limits(image,dir,c)

    band = image[y1:y2,x1:x2]
    edges = cv2.Canny(band,30,150,apertureSize = 3)   
    # edges = band < threshold_sauvola(band,25)
    # edges = cv2.Canny(band,30,150,apertureSize = 3)
    n = 2
    # kernel = np.ones((n,n))
    if dir == 'm':
        # binary = 255-black_white(band)
        binary = black_white(band)
    else:
        # binary = 255-black_white(band)
        if image.size > 1e7:
            # binary = band < threshold_niblack(band,121,k=-0.2)
            binary = black_white(band,threshold='niblack',window_size=121,k=-0.2)
            binary = cv2.blur(np.uint8(binary),(7,7))
        else:
            # binary = band < threshold_niblack(band,81,k=-0.2)
            binary = black_white(band,threshold='niblack',window_size=81,k=-0.2)

    d = e = f = None

    if verbose:
        _, (ax,e,f,d) = plt.subplots(1,4,figsize=(16,5),width_ratios=[5,2,2,2])
        ax.imshow(image,cmap='gray')
        ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', fill=False))
        d.imshow(band,cmap='gray')
        e.imshow(edges,cmap='gray')
        f.imshow(binary,cmap='gray')
        ax.autoscale(False)
        d.autoscale(False)
        e.autoscale(False)
        f.autoscale(False)
        # plt.autoscale(False)

    a, b, x0, y0, theta = hough_vertical_lines(edges,angle,nbmax=nbmax)
    all_rbw, all_x, all_y, all_risk, all_angle, all_cont = get_good_lines(band,binary,x1,ratio,continuous_ratio,a,b,x0,y0,theta,d=d,e=e,f=f)   
    if verbose:
        print('rbw',all_rbw, 'risk', all_risk, 'cont', all_cont)
    
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
        # i = np.argmax(all_rbw)
        i = np.argmin(all_risk) # get index of least risky split (splitted zone are more homogeneous)
    
    print(f'split at x={all_x[i]} and angle={round(all_angle[i],5)} with coverage of {round(all_rbw[i],5)} and risk of {round(all_risk[i],0)} and continuous ratio of {round(all_cont[i],5)}')

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

def all_orientation(img,ratio=0.6,continuous_ratio=0.45,c=4,angle=np.pi/9,verbose=True,cvd=False):
    '''
    '''
    x = np.array([])
    y = np.array([])
    a = np.array([])
    for dir in ['l','r','u','d']:
        x_,y_,a_ = get_split(img,dir,ratio,continuous_ratio,c,angle,verbose,cvd)
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
        # print(sim)
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

def rectify(img,x=None,y=None,ang=None,rad=None,verbose=False):
    '''
    Must provide rad or ang
    '''
    height, width = img.shape[:2]
    center = (width/2,height/2)
    if ang is not None:
        rad = main_orientation(ang)
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
    
    if x is None or y is None:
        return rotated_image
    
    x_1 = (x*rotate_matrix[0,0] + y*rotate_matrix[0,1] + rotate_matrix[0,2]).astype(int)
    y_1 = (x*rotate_matrix[1,0] + y*rotate_matrix[1,1] + rotate_matrix[1,2]).astype(int)
      
    x_2 = (y*rotate_matrix[0,0] + x*rotate_matrix[0,1] + rotate_matrix[0,2]).astype(int)
    y_2 = (y*rotate_matrix[1,0] + x*rotate_matrix[1,1] + rotate_matrix[1,2]).astype(int)

    x_ = np.concatenate((x_1[:2],x_2[2:]))
    y_ = np.concatenate((y_1[:2],y_2[2:]))
        
    if verbose and ang is not None:
        fig, (a,b) = plt.subplots(1,2,figsize=(15,10))

        a.imshow(img)
        a.autoscale(False)
        a_ = -ang - np.pi/2
        s = -np.sin(a_)/np.cos(a_)
        a.axline((x_[0],y_[0]),slope=-s[0],color='skyblue',linestyle='dashed')
        a.axline((x_[1],y_[1]),slope=-s[1],color='skyblue',linestyle='dashed')
        a.axline((x_[2],y_[2]),slope=1/s[2],color='skyblue',linestyle='dashed')
        a.axline((x_[3],y_[3]),slope=1/s[3],color='skyblue',linestyle='dashed')
        a.scatter(x[:2],y[:2],marker='x',color='red')
        a.scatter(y[2:],x[2:],marker='x',color='red')
        
        b.imshow(rotated_image)
        b.autoscale(False)
        b.axvline(x_[0],color='tomato',linestyle='dashed')
        b.axvline(x_[1],color='tomato',linestyle='dashed')
        b.axhline(y_[2],color='tomato',linestyle='dashed')
        b.axhline(y_[3],color='tomato',linestyle='dashed')
        b.scatter(x_,y_,marker='x',color='red')

        plt.show()

    return rotated_image[y_[2]:y_[3],x_[0]:x_[1]]

def split(img):
    pass

def reframe(img, rep=2, trim_required=True, save_path=None, verbose=False):
    '''
    '''
    x,y,a = get_split(img,'m',c=9,continuous_ratio=0.45,ratio=0.8, verbose=verbose)
    if x is not None:
        ends = (int((len(img)-y)*np.tan(a) + x),int(-y*np.tan(a) + x))
        imgs = [img[:,:max(ends)], img[:,min(ends):]]
    else:
        imgs = [img]
    if verbose:
        for i in range(len(imgs)):
            plt.imshow(imgs[i])
            plt.yticks([])
            plt.xticks([])
            plt.show()

    if save_path is not None:
        j = len(save_path)-save_path[::-1].index('.')-1
        k = len(save_path)-save_path[::-1].index('/')
        if not os.path.exists(save_path[:k]+'reframed/'):
            os.makedirs(save_path[:k]+'reframed/')
        for i in range(len(imgs)):
            print(imgs[i].shape)
            ind_sp = ''.join((save_path[:k],'reframed/',save_path[k:j],'_'+str(i)+'0','.png'))
            plt.imsave(ind_sp,imgs[i])
        
    if trim_required:
        for i in range(len(imgs)):
            for r in range(rep):
                x,y,a = all_orientation(imgs[i],continuous_ratio=0.45,verbose=verbose)
                imgs[i] = rectify(imgs[i],x,y,a,verbose)
        if save_path is not None:
            j = len(save_path)-save_path[::-1].index('.')-1
            k = len(save_path)-save_path[::-1].index('/')
            if not os.path.exists(save_path[:k]+'reframed/'):
                os.makedirs(save_path[:k]+'reframed/')
            for i in range(len(imgs)):
                print(imgs[i].shape)
                ind_sp = ''.join((save_path[:k],'reframed/',save_path[k:j],'_'+str(i)+'1','.png'))
                plt.imsave(ind_sp,imgs[i])

    _, axes = plt.subplots(1, len(imgs)+1, figsize=(10 + 10*len(imgs),10))
    axes[0].imshow(img)
    for i in range(1,len(axes)):
        axes[i].imshow(imgs[i-1])
    plt.show()

    return imgs

def reframe_(img,rep=2,save_path=None):
    '''
    '''
    x,y,a = get_split(img,'m',c=9,continuous_ratio=0.45,ratio=0.3)
    if x is not None:
        ends = (int((len(img)-y)*np.tan(a) + x),int(-y*np.tan(a) + x))
        imgs = [img[:,:max(ends)], img[:,min(ends):]]
    else:
        imgs = [img]
    
    for i in range(len(imgs)):
        for r in range(rep):
            x,y,a = all_orientation(imgs[i],continuous_ratio=0.45)
            imgs[i] = rectify(imgs[i],x,y,a)

    if save_path is not None:
        j = len(save_path)-save_path[::-1].index('.')-1
        k = len(save_path)-save_path[::-1].index('/')
        if not os.path.exists(save_path[:k]+'reframed/'):
            os.makedirs(save_path[:k]+'reframed/')
        for i in range(len(imgs)):
            print(imgs[i].shape)
            ind_sp = ''.join((save_path[:k],'reframed/',save_path[k:j],'_'+str(i),'.png'))
            plt.imsave(ind_sp,imgs[i])

    return imgs
