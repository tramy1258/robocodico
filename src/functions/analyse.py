import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import median_abs_deviation
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from math import floor
from functions.utils import show_linked_areas

colors = ['orchid','tomato','olive','cadetblue','cornflowerblue','goldenrod','darkseagreen','crimson','lightpink']
# def count_text(l):
    # '''
    # Count number of consecutive 0s or 1s in given list l, for example [0 1 0 0 0 1 1] gives [1 1 3 2]
    # gives [0 2 7] [1 1 0 0 1 1 1] [2 7]
    # '''
    # res = []
    # curval = -1
    # for i in range(len(l)):
    #     if l[i] != curval:
    #         if l[i] == 0:
    #             res.append(i)
    #         curval = l[i]
            
    # return res + [len(l)]

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

# def check_text1(areas,imgs,txtstd=5,verbose=True):
#     '''
#     '''
#     for i in range(len(areas)):
#         print('IMAGE',i)
#         text_areas = []

#         if len(imgs[i].shape) == 3:
#             new_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
#         else:
#             new_img = imgs[i].copy()

#         for x1,y1,x2,y2 in areas[i]:
#             print('\nAnalyzing area',x1,y1,x2,y2)
#             img = new_img[y1:y2,x1:x2]
#             if verbose:
#                 _,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,4),width_ratios=(3,6,6))
#                 ax1.imshow(img,cmap='gray')
#                 ax2.plot(np.mean(img,1))
#                 ax2.axhline(np.mean(img),color='tomato')
#                 ax3.plot(np.mean(img,0))
#                 plt.show()
#             # compute periods
#             ud = np.where(np.mean(img,1) > np.mean(img),1,0)
#             periods = count_up_down(ud)
#             print(periods[::2],periods[1::2])
#             std_text = np.std(periods[::2]) if ud[0] == 0 else np.std(periods[1::2])
#             std_blank = np.std(periods[1::2]) if ud[0] == 0 else np.std(periods[::2])
#             print(std_text,std_blank)

#             # check if periodic
#             if std_blank < txtstd and std_text < txtstd and len(periods) > 3:
#                 nb_lines = len(periods[::2]) if ud[0] == 0 else len(periods[1::2])
#                 print(f'----->>> Contains {nb_lines} lines of text')
#                 text_areas.append((x1,y1,x2,y2))
#             else:
#                 print('---> not text')
#             new_img[y1:y2,x1:x2] = np.mean(img)

#         _,(ax1,ax2) = plt.subplots(1,2,figsize=(9,4))
#         ax1.imshow(new_img,cmap='gray')
#         ax2.imshow(imgs[i])
#         for x1,y1,x2,y2 in text_areas:
#             rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='tomato', fill=False)
#             ax1.add_patch(rect)
#             rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='tomato', fill=False)
#             ax2.add_patch(rect)
#         plt.show()
    
# def check_partial_text():
#     pass

def check_text(areas,imgs,text_std=6,verbose=True):
    '''
    '''
    all_text_areas = []
    for i in range(len(areas)):
        print('IMAGE',i)
        text_areas = []

        # if len(imgs[i].shape) == 3:
        #     new_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        # else:
        #     new_img = imgs[i].copy()

        for a in areas[i]:
            print('Analyzing area',a)
            nblines = analyze_text(imgs[i],*a,text_std)

            if nblines != -1:
                text_areas.append((*a,nblines))
            else:
                print('---> not text')

            # new_img[y1:y2,x1:x2] = np.mean(img)

        all_text_areas.append(text_areas)
        linked_texts = link(text_areas)

        _,(ax2,ax1) = plt.subplots(1,2,figsize=(9,7))
        ax2.imshow(imgs[i])
        for a in text_areas:
            x1,y1,x2,y2 = a[:4]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='tomato', fill=False)
            ax2.add_patch(rect)
        show_linked_areas(imgs[i],linked_texts,ax=ax1)
        plt.show()
        
        print('This page contains',len(linked_texts),'columns of text.')
        for j,a in linked_texts.items():
            print('- column',j,'in',colors[j],'contains',a[1],'lines of text')

    print(all_text_areas)
    return all_text_areas
    
def get_minima(l):#,v=-1):
    # if v==-1:
    mins = [0] + [1 if l[i]<l[i-1] and l[i]<l[i+1] else 0 for i in range(1,len(l)-1)]
    # else:
        # mins = [0] + [1 if l[i]<l[i-1] and l[i]<l[i+1] and l[i]<v else 0 for i in range(1,len(l)-1)]
    mins = [i for i in range(len(mins)) if mins[i]]
    return mins

def outer_module(img,ax=None):
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = []
    hmean = np.mean(img,1)
    l = np.where(hmean < np.mean(img),0,1)

    for i in range(len(l)-1):
        if l[i] != l[i+1]:
            res.append(i)

    res = [[res[i],res[i+1]] for i in range(l[res[0]+1],len(res)-1,2)]
    if ax is not None:
        for t in res:
            ax.axhline(t[0],color='teal')
            ax.axhline(t[1],color='teal')
    return res

def inner_module(img,out=None,ax=None):
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = []
    hmean = np.mean(img,1)
    # color = ['red','salmon','palegreen','cornflowerblue']
    if out is None:
        out = outer_module(img)
    for t in out:
        mins = [m+t[0] for m in get_minima(hmean[t[0]:t[1]])]
        mins = [min(mins),max(mins)] if len(mins) > 0 else []
        res.append(mins)
        if ax is not None:
            for t in mins:
                ax.axhline(t,color='tomato')
                # ax.axhline(t,color=color[(len(mins)-1)%len(color)])
    return res

def analyze_text(image,x1,y1,x2,y2,text_std):
    if image.ndim > 2:
        text_img = cv2.cvtColor(image[y1:y2,x1:x2], cv2.COLOR_RGB2GRAY)
    else: 
        text_img = image[y1:y2,x1:x2]
    
    _,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(23,15),width_ratios=(7,8,8))
    # ax3.imshow(image[y1:y2,x1:x2])
    ax3.plot(np.mean(text_img,0),color='darkseagreen')
    ax3.axhline(np.mean(text_img),color='goldenrod')
    ax2.plot(np.mean(text_img,1),color='cornflowerblue')
    ax2.axhline(np.mean(text_img),color='lightsalmon')
    # text_img = text_img > threshold_sauvola(text_img,25)
    # ax4.imshow(text_img,cmap='gray')
    ax1.imshow(image[y1:y2,x1:x2])
    # ax2.plot(np.mean(text_img,1)*255,color='blue')
    # ax2.axhline(np.mean(text_img)*255,color='red')
    ax1.autoscale(False)
    
    text = outer_module(text_img,ax=ax1)
    text_ = inner_module(text_img,ax=ax1,out=text)
    # print(text)
    # print(text_)

    plt.show()

    outer_text_size = [t[1]-t[0] for t in text if len(t)==2]
    inner_text_size = [t[1]-t[0] for t in text_ if len(t)==2]
    blank_size = [text[i+1][0]-text[i][1] for i in range(len(text)-1)]
    # blank_size = [b for b in blank_size if b < 2*np.median(outer_text_size)]

    # or len(inner_text_size) == 0 or np.std(inner_text_size) > text_std 
    print(len(outer_text_size), len(blank_size), np.std(outer_text_size), np.std(blank_size))
    if len(outer_text_size) == 0 or len(blank_size) == 0\
        or np.std(outer_text_size) > text_std or np.std(blank_size) > text_std\
        or any([b > 2*np.median(outer_text_size) for b in blank_size]):
        return -1 

    inked = np.median(outer_text_size) #+0.5
    blank = np.median(blank_size)
    print('outer module size',outer_text_size,'\n median =', np.median(outer_text_size),'mean =',np.mean(outer_text_size),'std =', np.std(outer_text_size))
    # print('inner module size',inner_text_size,'\n median =', np.median(inner_text_size),'mean =',np.mean(inner_text_size),'std =', np.std(inner_text_size))
    print('interline size',blank_size,'\n median =', np.median(blank_size),'mean =',np.mean(blank_size),'std =', np.std(blank_size))
    
    print((y2-y1-inked)/(inked+blank))
    nblines = floor((y2-y1-inked)/(inked+blank)) # + 1
    print(nblines,'lines of text')
    
    return nblines

# def same_length_neighbor(a,b):
#     x1,y1,x2,y2 = a[:4]
#     for a_ in b:
#         x3,y3,x4,y4 = a_[:4]
#         if ((x1==x4 or x2==x3) and y1 == y3 and y2 == y4):
#             return True
#     return False

def left_right_neighbor(a,b):
    x1,y1,x2,y2 = a[:4]
    for a_ in b:
        x3,y3,x4,y4,nb = a_
        if x1==x4 or x2==x3:
            if (min(y1,y2)-max(y3,y4)) * (min(y3,y4)-max(y1,y2)) > 0:
                return nb
    return 0

def up_down_neighbor(a,b):
    x1,y1,x2,y2 = a[:4]
    for a_ in b:
        x3,y3,x4,y4,nb = a_
        if y1==y4 or y2==y3:
            if (min(x1,x2)-max(x3,x4)) * (min(x3,x4)-max(x1,x2)) > 0:
                return 1
    return 0

def link(areas):
    linked = dict()
    for i in range(len(areas)):
        print(areas[i])
        have_neighbor = False
        for c,b in linked.items():
            print('-'*50)
            nb = left_right_neighbor(areas[i],b[0])
            if nb:
                print('a',nb)
                have_neighbor = True
                b[0].append(areas[i])
                linked[c] = (b[0],max(nb,areas[i][4]))
                continue
            nb = up_down_neighbor(areas[i],b[0])
            if nb:
                print('b',nb)
                have_neighbor = True
                b[0].append(areas[i])
                linked[c] = (b[0],b[1]+areas[i][4])
                continue
        if not have_neighbor:
            linked[i] = ([areas[i]],areas[i][4])
        print(linked)
    print(linked)
    return linked

def get_words(img,x1,y1,x2,y2,over=50,delta=6,blankwidth=0,verbose=False):
    if img.ndim > 2:
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        image = img
    text_img = image[y1+blankwidth-delta:y2+delta,x1:x2+over]
    text_img = text_img < threshold_sauvola(text_img,25,k=0.2)
    text_img_blurred = cv2.dilate(np.uint8(text_img),np.ones((2,2)),iterations=3)
    text_img_blurred = cv2.blur(np.uint8(text_img_blurred),(3,3))
    text_img_colored = img[y1+blankwidth-delta:y2+delta,x1:x2+over]

    text = outer_module(text_img[:,0:x2-x1])
    overboarding = []
    for t in text[:]:
        line = text_img[max(t[0]-delta,0):min(t[1]+delta,text_img.shape[0])]
        line_blurred = text_img_blurred[max(t[0]-delta,0):min(t[1]+delta,text_img_blurred.shape[0])]
        line_colored = text_img_colored[max(t[0]-delta,0):min(t[1]+delta,text_img_colored.shape[0])]
        # line = text_img[t[0]:t[1]]
        # line_blurred = text_img_blurred[t[0]:t[1]]
        # line_colored = text_img_colored[t[0]:t[1]]
        # plt.figure(figsize=(50,5))
        blank = True
        nblack = 0

        for c in range(line.shape[1]-1,-1,-1):
            if not blank:
                break
            if np.sum(line[:,c]) == 0:
                nblack+=1
            else:
                blank = False
        
        overboarding.append((x1, y1-delta+max(t[0]-delta,0),
                            x2+over-nblack, y1-delta+min(t[1]+delta,text_img_blurred.shape[0])))
        if verbose:
            print(t,max(t[0]-delta,0),min(t[1]+delta,text_img.shape[0]))
            print('Text overboarding for',over-nblack,'pixels.')
            fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,figsize=(62,7))
            ax1.imshow(line,cmap='gray')
            ax2.imshow(line_blurred,cmap='gray')
            ax3.imshow(line_colored)
            ax4.imshow(line_colored)
            ax5.imshow(line_colored)
            ax6.imshow(line,cmap='gray')
            # ax1.autoscale(False)
            # ax2.autoscale(False)
            # ax3.autoscale(False)
            # ax4.autoscale(False)
            # ax5.autoscale(False)
            # ax6.autoscale(False)

            for c in range(line.shape[1]):
                if np.sum(line[:,c]) == 0:
                    ax1.axvline(c,color='red',linewidth=3)
                    ax3.axvline(c,color='red',linewidth=3)
            for c in range(line_blurred.shape[1]):
                if np.sum(line_blurred[:,c]) == 0:
                    ax2.axvline(c,color='teal',linewidth=3)
                    ax4.axvline(c,color='teal',linewidth=3)#,linestyle='dashed')

            ax1.axvline(x2-x1,color='gold',linewidth=3)
            ax2.axvline(x2-x1,color='gold',linewidth=3)
            ax3.axvline(x2-x1,color='darkkhaki',linewidth=3)
            ax4.axvline(x2-x1,color='darkkhaki',linewidth=3)
            ax5.axvline(x2-x1,color='darkkhaki',linewidth=3)
            ax6.axvline(x2-x1,color='gold',linewidth=3)
            plt.show()

    return overboarding
    
def delete_text_binary(img,text_areas,over=30,delta=6,verbose=False):
    if img.ndim > 2:
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        image = img
    image = image < threshold_sauvola(image,25,k=0.2)
    if verbose:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,10))
        ax1.imshow(image,cmap='gray')
    for x1,y1,x2,y2,nb in text_areas:
        image[y1:y2,x1:x2] = 0
        overboarding = get_words(img,x1,y1,x2,y2,over,delta)
        # print(overboarding)
        for x,y,x_,y_ in overboarding:
            image[y:y_,x:x_] = 0
            if verbose:
                rect = patches.Rectangle((x, y), x_-x, y_-y, linewidth=1, edgecolor='tomato', fill=False)
                ax1.add_patch(rect)
    
    if verbose:
        ax2.imshow(image,cmap='gray')
        plt.show()
    return image

def get_neighbors(binary, pixel, eps):
    i, j = pixel
    neighborhood = np.array([[x,y] for x in range(max(i-eps,0),min(i+eps,len(binary[0]))) 
                                   for y in range(max(j-eps,0),min(j+eps,len(binary)))
                                   if binary[y,x]
                            ])
    return neighborhood

def get_white_pixels_coords(binary, eps):
    #todo: graph (adjacent lists)
    list_white_pixels = np.array([[x,y] for x in range(len(binary[0])) 
                                        for y in range(len(binary)) 
                                        if binary[y,x]
                                ])
    graph_white_pixels = {px: get_neighbors(binary, px, eps) for px in list_white_pixels}
    state_white_pixels = {px: -1 for px in list_white_pixels}
    return state_white_pixels, graph_white_pixels
        
# def get_connected_components_knn(binary):
#     white_pixels = get_white_pixels_coords(binary)
#     connected = [-1]*white_pixels.shape[0]
#     connected_dict = dict()
#     cpt = 0
#     for i in range(white_pixels.shape[0]):
#         dist = np.argsort(np.sum((white_pixels-white_pixels[i])**2,axis=1))
#         j = dist[1]
#         if connected[i] != -1 and connected[j] != -1:
#             connected_dict[connected[i]] |= connected_dict[connected[j]]
#         elif connected[i] != -1:
#             connected[j] = connected[i]
#             connected_dict[connected[i]].add(j)
#         elif connected[j] != -1:
#             connected[i] = connected[j]
#             connected_dict[connected[j]].add(i)
#         else:
#             cpt+=1
#             connected[j] = cpt
#             connected[i] = connected[j]
#             connected_dict[cpt] = {i,j}
#     # print(connected_dict)
#     for cluster in connected_dict:
#         for i in connected_dict[cluster]:
#             connected[i] = cluster
#     plt.figure(figsize=(10,15))
#     plt.imshow(binary,cmap='gray')
#     plt.scatter(white_pixels[:,0],white_pixels[:,1],c=np.array(connected),cmap='Paired',s=1)
#     plt.show()
#     print(cpt,'clusters found with \'1 nearest neighbour\' algorithm')
#     values, counts = np.unique(connected,return_counts=True)
#     for i in range(len(values)):
#         print('- cluster',values[i],'has',counts[i],'elements.')
#     return connected

def get_connected_components_dbscan(binary, eps=10, minpts=15, colored=None):
    white_pixels = get_white_pixels_coords(binary)
    connected = [-1]*white_pixels.shape[0]
    res = []
    cpt = 0
    for i in range(white_pixels.shape[0]):
        if connected[i] != -1:
            continue
        dist = np.sqrt(np.sum((white_pixels-white_pixels[i])**2,axis=1))
        neighbors = {j for j in range(white_pixels.shape[0]) if dist[j] > 0 and dist[j] < eps}
        if len(neighbors) < minpts:
            connected[i] = 0
            continue
        cpt += 1
        connected[i] = cpt
        nb_neighbors = len(neighbors)

        while len(neighbors) > 0:
            j = neighbors.pop()
            if connected[j] == 0:
                connected[j] = cpt

            if connected[j] != -1:
                continue
                
            connected[j] = cpt

            dist = np.sqrt(np.sum((white_pixels-white_pixels[j])**2,axis=1))
            neighbors_of_neighbor = {k for k in range(white_pixels.shape[0]) if dist[k] > 0 and dist[k] < eps}
            
            if len(neighbors_of_neighbor) >= minpts:
                neighbors |= neighbors_of_neighbor
    if colored is not None:
        fig, (ax,ax_) = plt.subplots(1,2,figsize=(20,15))
        ax_.imshow(colored)
    else:
        fig, ax = plt.subplots(1,1,figsize=(10,15))
    ax.imshow(binary,cmap='gray')
    ax.scatter(white_pixels[:,0],white_pixels[:,1],c=np.array(connected),cmap='Paired',s=1)
    print(cpt,'clusters found with DBSCAN algorithm')
    values, counts = np.unique(connected,return_counts=True)
    for i in range(len(values)):
        if values[i] == 0:
            continue
        print('- cluster',values[i],'covers',counts[i],'pixel.')
        indices = np.array([white_pixels[j] for j in range(white_pixels.shape[0]) if connected[j] == values[i]])
        x = np.min(indices[:,0])
        x_ = np.max(indices[:,0])
        y = np.min(indices[:,1])
        y_ = np.max(indices[:,1])
        res.append((x,y,x_,y_))
        rect = patches.Rectangle((x, y), x_-x, y_-y, linewidth=1, edgecolor='tomato', fill=False)
        ax.add_patch(rect)
        if colored is not None:
            rect = patches.Rectangle((x, y), x_-x, y_-y, linewidth=1, edgecolor='tomato', fill=False)
            ax_.add_patch(rect)
    plt.show()
    return res


def highlight_fond_color(image,color='tomato',i=None):
    '''
    Parameters:
        - image: 2D array
    '''
    hist,vals = np.histogram(image,bins=50)
    plt.hist(image.flatten(),bins=50)
    plt.show()
    print('hist =',hist,np.sum(hist),len(hist))
    print('vals =',vals,len(vals))
    if i is None:
        i = np.argmax(hist)
    print(vals[i],vals[i+1])
    xy = [(x,y) for y in range(len(image)) for x in range(len(image[0])) if (image[y,x]>=vals[i] and image[y,x]<vals[i+1])]
    xy = np.array(xy)
    plt.figure(figsize=(20,10))
    plt.imshow(image,cmap='gray')
    plt.scatter(xy[:,0],xy[:,1],color=color,marker='.',s=1)
    plt.show()

def otsu_multiple_gaussians(image, n=2, bins=50):
    hist, vals = np.histogram(image,bins=bins)
    limit = -1

    for i in range(n):
        print('-'*50)
        print(limit)
        print(len(hist),len(vals))
        hist = hist[:limit+1] if limit != -1 else hist[:]
        vals = vals[:limit+2] if limit != -1 else vals[:]
        
        bin_mids = (vals[:-1] + vals[1:]) / 2.
        print(len(hist),len(bin_mids),len(vals))
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        mean1 = np.cumsum(hist * bin_mids) / weight1
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
        
        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        index_of_max_val = np.argmax(inter_class_variance)
        threshold = bin_mids[:-1][index_of_max_val]

        limit = index_of_max_val
        ax = plt.subplot(111)
        ax.bar(vals[:-1],hist,width=5,color='skyblue')
        ax_ = ax.twinx()
        ax_.plot(vals[1:-1],inter_class_variance,color='tomato')
        ax_.axvline(threshold,linestyle='dashed',color='tomato')
        plt.show()
    # plt.show()

