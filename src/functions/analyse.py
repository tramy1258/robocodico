import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import median_abs_deviation

def count_text(l):
    '''
    Count number of consecutive 0s or 1s in given list l, for example [0 1 0 0 0 1 1] gives [1 1 3 2]
    gives [0 2 7] [1 1 0 0 1 1 1] [2 7]
    '''
    res = []
    curval = -1
    #for i in l:
    #    if i == curval:
    #        res[-1] += 1
    #    else:
    #        res.append(1)
    #        curval = i
    for i in range(len(l)):
        if l[i] != curval:
            if l[i] == 0:
                res.append(i)
            curval = l[i]
            
    return res + [len(l)]

def count_up_down(l):
    '''
    Count number of consecutive 0s or 1s in given list l, for example [0 1 0 0 0 1 1] gives [1 1 3 2]
    gives [0 2 7] [1 1 0 0 1 1 1] [2 7]
    '''
    res = []
    curval = -1
    for i in l:
        if i == curval:
            res[-1] += 1
        else:
            res.append(1)
            curval = i
    return res

def check_text(areas,imgs,txtstd=5,verbose=True):
    '''
    '''
    for i in range(len(areas)):
        print('IMAGE',i)
        text_areas = []

        if len(imgs[i].shape) == 3:
            new_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        else:
            new_img = imgs[i].copy()

        for x1,y1,x2,y2 in areas[i]:
            print('\nAnalyzing area',x1,y1,x2,y2)
            img = new_img[y1:y2,x1:x2]
            if verbose:
                _,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,4),width_ratios=(3,6,6))
                ax1.imshow(img,cmap='gray')
                ax2.plot(np.mean(img,1))
                ax2.axhline(np.mean(img),color='tomato')
                ax3.plot(np.mean(img,0))
                plt.show()
            # compute periods
            ud = np.where(np.mean(img,1) > np.mean(img),1,0)
            periods = count_up_down(ud)
            print(periods[::2],periods[1::2])
            std_text = np.std(periods[::2]) if ud[0] == 0 else np.std(periods[1::2])
            std_blank = np.std(periods[1::2]) if ud[0] == 0 else np.std(periods[::2])
            print(std_text,std_blank)

            # check if periodic
            if std_blank < txtstd and std_text < txtstd and len(periods) > 3:
                nb_lines = len(periods[::2]) if ud[0] == 0 else len(periods[1::2])
                print(f'----->>> Contains {nb_lines} lines of text')
                text_areas.append((x1,y1,x2,y2))
            else:
                print('---> not text')
            new_img[y1:y2,x1:x2] = np.mean(img)

        _,(ax1,ax2) = plt.subplots(1,2,figsize=(9,4))
        ax1.imshow(new_img,cmap='gray')
        ax2.imshow(imgs[i])
        for x1,y1,x2,y2 in text_areas:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='tomato', fill=False)
            ax1.add_patch(rect)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='tomato', fill=False)
            ax2.add_patch(rect)
        plt.show()
    
def check_partial_text():
    pass


def check_text1(areas,imgs,txtstd=5,verbose=True):
    '''
    '''
    for i in range(len(areas)):
        print('IMAGE',i)
        text_areas = []

        if len(imgs[i].shape) == 3:
            new_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        else:
            new_img = imgs[i].copy()

        for x1,y1,x2,y2 in areas[i]:
            print('\nAnalyzing area',x1,',',y1,',',x2,',',y2)
            img = new_img[y1:y2,x1:x2]
            if verbose:
                _,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,4),width_ratios=(3,6,6))
                ax1.imshow(img,cmap='gray')
                ax2.plot(np.mean(img,1))
                ax2.axhline(np.mean(img),color='tomato')
                ax3.plot(np.mean(img,0))
                
            # compute periods
            ud = np.where(np.mean(img,1) > np.mean(img),1,0)
            periods = count_text(ud)
            print(periods)
            text = [np.argmin(np.mean(img,1)[periods[i]:periods[i+1]]) + periods[i] for i in range(len(periods)-1)]
            for t in text:
                ax1.axhline(t,color='tomato')
            plt.show()
            text_periods = [text[i+1]-text[i] for i in range(len(text)-1)]
            std_text = np.std(text_periods)
            print('text =',text_periods,'stdev =',std_text)
            print('vertical median_abs_deviation',median_abs_deviation(np.mean(img,0)))

            # check if periodic
            if std_text < txtstd and len(text_periods) > 1:
                nb_lines = len(text)
                print(f'----->>> Contains {nb_lines} lines of text')
                text_areas.append((x1,y1,x2,y2))
            else:
                print('---> not text')
            new_img[y1:y2,x1:x2] = np.mean(img)

        _,(ax1,ax2) = plt.subplots(1,2,figsize=(9,4))
        ax1.imshow(new_img,cmap='gray')
        ax2.imshow(imgs[i])
        for x1,y1,x2,y2 in text_areas:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='teal', fill=False)
            ax1.add_patch(rect)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='tomato', fill=False)
            ax2.add_patch(rect)
        plt.show()
    
def get_minima(l,v=-1):
    if v==-1:
        mins = [0] + [1 if l[i]<l[i-1] and l[i]<l[i+1] else 0 for i in range(1,len(l)-1)]
    else:
        mins = [0] + [1 if l[i]<l[i-1] and l[i]<l[i+1] and l[i]<v else 0 for i in range(1,len(l)-1)]
    mins = [i for i in range(len(mins)) if mins[i]]
    return mins

def outer_module(img,ax=None):
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = []
    hmean = np.mean(img,1)
    l = np.where(hmean > np.mean(img),1,0)
    if l[0] == 0:
        res.append(0)
    for i in range(1,len(l)):
        if l[i] != l[i-1]:
            res.append(i)
    if l[-1] == 0 and res[-1] != len(l):
         res.append(len(l))
    if ax is not None:
        for t in res:
            ax.axhline(t,color='teal')
    return [[res[i],res[i+1]] for i in range(0,len(res),2)]

def inner_module(img,out=None,ax=None):
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = []
    hmean = np.mean(img,1)
    color = ['red','salmon','palegreen','cornflowerblue']
    if out is None:
        out = outer_module(img)
    for t in out:
        mins = [m+t[0] for m in get_minima(hmean[t[0]:t[1]])]
        res.append(mins)
        if ax is not None:
            for t in mins:
                ax.axhline(t,color=color[(len(mins)-1)%len(color)])
    return res

def analyze_text(img,x1,y1,x2,y2,gray=False):
    if img.ndim > 2 and gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    text_img = img[y1:y2,x1:x2]
    _,(ax1,ax2) = plt.subplots(1,2,figsize=(22,15),width_ratios=(20,30))
    ax1.imshow(text_img,cmap='gray')
    ax2.plot(np.mean(text_img,1),color='cornflowerblue')
    ax2.axhline(np.mean(text_img),color='tomato')
    text = inner_module(text_img,ax=ax1)
    size = [t[1]-t[0] for t in text if len(t)==2]
    # print('text limits',text)
    print('inner module size',size,'\n mean =', np.mean(size),'std =', np.std(size))
    text = outer_module(text_img,ax=ax1)
    size = [t[1]-t[0] for t in text if len(t)==2]
    # print('text limits',text)
    print('outer module size',size,'\n mean =', np.mean(size),'std =', np.std(size))
    # print('inner_module',inner_module(text_img,ax=ax1))
    plt.show()

