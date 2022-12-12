import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from functions.utils import show_splits

def ponderated_criterion(image,criterion,split,axis):
    """ - axis can only be 'x' or 'y' 
        - normalized gray image only 
        Return ponderated sum of a certain criterion on splitted image  
    """

    if axis == 'x':
        left  = image[:,:split]
        right = image[:,split:]
        if left.size == 0 or right.size == 0: # impossible position
            print('extreme')
            return image.size # if m == 1 else 0
        pond  = criterion(left,np.ones_like(left)+np.mean(left)-1) * left.size + criterion(right,np.ones_like(right)+np.mean(right)-1) * right.size
        #pond /= image.size
    else:
        left  = image[:split]
        right = image[split:]
        if left.size == 0 or right.size == 0: # impossible position
            print('extreme')
            return image.size # if m == 1 else 0
        pond = criterion(left,np.ones_like(left)+np.mean(left)-1) * left.size + criterion(right,np.ones_like(right)+np.mean(right)-1) * right.size
        #pond /= image.size
    return pond
    
def best_split_general(image,criterion,ax3,x_start=0,y_start=0,rate=1,uni_axis=None,no_split=None):
    """ - normalized gray image only 
        - m=1 for minimizing and -1 for maximizing
        Return best split for an image under a certain criterion, considering both directions
    """
    if len(image.shape) != 2: # or m not in [1,-1]:
        print('Problem with m or image')
        return

    all_splits = []
    if no_split is None:
        no_split = criterion(image,np.ones_like(image)+np.mean(image)-1)
    for dir in [('x',len(image[0]),ax3,x_start),('y',len(image),ax3,y_start)]:
        axis, t_pos, ax, start = dir
        if uni_axis is not None and axis != uni_axis:
            continue

        all_risks = [(axis,t,ponderated_criterion(image,criterion,t,axis)) for t in range(1,t_pos,rate)]
        all_risks = np.array(all_risks)
        #print(all_risks)

        if len(all_risks) == 0:
            all_splits.append((axis,-1,2))
            continue
        ax.plot(np.arange(1,t_pos,rate)+start,[no_split-float(c) for c in all_risks[:,2]])
        ax.set(yticklabels=[])
        all_splits.append(all_risks[np.argmin([float(r[2]) for r in all_risks])])
    all_splits = np.array(all_splits)
    return all_splits[np.argmin([float(r[2]) for r in all_splits])]

def cart_regression_1(image,criterion,x_start=0,y_start=0,max_depth=10,eps_=1e-5,rate=1,whole_image=None,done_splits=[],best=None):
    """ Split image recursively
    """
    print('DEPTH =',max_depth)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 3))
    plt.yticks([])
    if whole_image is not None:
        show_splits(whole_image,done_splits,ax1,'olive')
    ax2.imshow(image,cmap='gray')
    splits = []
    #final_splits = []
    #print('size',image.size, len(image), len(image[0]))
    print(len(image),len(image[0]))
    if image.size == 0 or max_depth <= 0:
        #return final_splits
        return splits
    
    # get best splitting position and direction 
    #print(best_split(image, criterion))
    if best is None:
        best = best_split_general(image, criterion, ax3, x_start, y_start, rate=rate)
    if best is None:
        #return final_splits
        return splits
    axis,t,emp_risk_split = best
    
    # see if splitting is neccessary ??
    emp_risk = criterion(image,np.ones_like(image)+np.mean(image)-1)
    ax3.axhline(emp_risk,color='olive')
    t = int(t)
    emp_risk_split = float(emp_risk_split)
    #print(emp_risk_split,emp_risk)
    if emp_risk_split < emp_risk and not np.abs(emp_risk_split - emp_risk) < eps_: # if neccessary and gives big difference
        if axis == 'x':
            ax1.axvline(t+x_start,linewidth=1,color='red')
            ax2.axvline(t,linewidth=1,color='red')
            plt.show()
            children  = [image[:,:t], image[:,t:]]
            risks = [criterion(im, np.ones_like(im)+np.mean(im)-1) for im in children] # risk of each splitted zone
            print('homogenity of each child',risks)
            print('ponderated homogenity of children if split original',emp_risk_split,'risk no split',emp_risk)

            best0 = best_split_general(children[0], criterion, ax3, x_start, y_start, rate=rate)
            if best0 is None:
                #return final_splits
                return splits
            axis0,t0,emp_risk_split0 = best0
            print(axis0,t0,emp_risk_split0)

            best1 = best_split_general(children[1], criterion, ax3, x_start+t, y_start, rate=rate)
            if best1 is None:
                #return final_splits
                return splits
            axis1,t1,emp_risk_split1 = best1
            print(axis1,int(t1)+t,emp_risk_split1)
            print('gain of splitting each child',-float(emp_risk_split0)+risks[0],-float(emp_risk_split1)+risks[1])
            bests =  [best0, best1]
            #print(risks - )
            # discard homogeneous zone, keep splitting hetero zone
            hm = np.argmin(risks)
            splits.append((axis,x_start+t))#,hm,risks[hm])) # hm = 0 for left/up, 1 for right/down
            x_start = x_start if hm else x_start+t
            splits += cart_regression_1(children[1-hm],criterion,x_start,y_start,max_depth-1,whole_image=whole_image,done_splits=done_splits+splits,best=bests[1-hm])
            #splits.append((axis,x_start+t))
            #final_splits.append((axis,x_start+t))
            #for c in range(2):
            #    final_splits += cart_regression(children[c],criterion,x_start + t*c,y_start,max_depth-1,whole_image=whole_image,done_splits=done_splits+splits)

        else: # if no
            ax1.axhline(t+y_start,linewidth=1,color='red')
            ax2.axhline(t,linewidth=1,color='red')
            plt.show()
            children  = [image[:t], image[t:]]
            risks = [criterion(im, np.ones_like(im)+np.mean(im)-1) for im in children] # risk of each splitted zone

            print('homogenity of each child',risks)
            print('ponderated homogenity of children if split original',emp_risk_split,'risk no split',emp_risk)
            best0 = best_split_general(children[0], criterion, ax3, x_start, y_start, rate=rate)
            if best0 is None:
                #return final_splits
                return splits
            axis0,t0,emp_risk_split0 = best0
            print(axis0,t0,emp_risk_split0)
            best1 = best_split_general(children[1], criterion, ax3, x_start+t, y_start+t, rate=rate)
            if best1 is None:
                #return final_splits
                return splits
            axis1,t1,emp_risk_split1 = best1
            print(axis1,int(t1)+t,emp_risk_split1)
            print('gain of splitting each child',-float(emp_risk_split0)+risks[0],-float(emp_risk_split1)+risks[1])
            bests =  [best0, best1]
            # discard homogeneous zone, keep splitting hetero zone
            hm = np.argmin(risks)
            splits.append((axis,y_start+t))#,hm,risks[hm])) # hm = 0 for left/up, 1 for right/down
            y_start = y_start if hm else y_start+t
            splits += cart_regression_1(children[1-hm],criterion,x_start,y_start,max_depth-1,whole_image=whole_image,done_splits=done_splits+splits,best=bests[1-hm])
            #splits.append((axis,y_start+t))
            #final_splits.append((axis,x_start+t))
            #for c in range(2):
            #    final_splits += cart_regression(children[c],criterion,x_start,y_start + t*c,max_depth-1,whole_image=whole_image,done_splits=done_splits+splits)
    else:
        plt.show()
    #return final_splits
    return splits

def cart_regression_uni(image,criterion,max_depth=10,eps_=8,rate=1):
    ''' Splits images iteratively and horizontally
    '''
    if image.size == 0 or max_depth <= 0:
        return []
    done_splits=[(0,None)] # (y_start, criterion of area from this y_start -> next y_start)
    chrono_splits=[]
    yet_splits=[None]
    best_t = None
    done_splits.append((len(image),0))
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize = (12, 3))
    plt.yticks([])
    ax1.imshow(image,cmap='gray')
    
    while (all([s is None for s in yet_splits]) or max_depth > 0):
        print('DEPTH =',max_depth)
        max_depth -= 1
        #print('done',done_splits,'yet',yet_splits,'chrono',chrono_splits)
        for i in range(len(done_splits)-1):
            if done_splits[i][1] is None:
                area = image[done_splits[i][0]:done_splits[i+1][0]]
                if area.size == 0:
                    return [('y',s[0]) for s in chrono_splits[1:]]
                done_splits[i] = (done_splits[i][0],criterion(area,np.ones_like(area)+np.mean(area)-1)*area.size)#,criterion(area,np.ones_like(area)+np.mean(area)-1))
        # area currently with s[0] splits, with risk s[1], considering new split at s[2] (verify if neccesary at next round)
        chrono_splits.append([len(done_splits)-2,np.sum([s[1] for s in done_splits])/1e6,None])
        ax2.scatter(np.arange(len(chrono_splits)),[s[1] for s in chrono_splits])
        plt.show()
        if len(chrono_splits) > 1:
            print('gain in empirical risk with splitting is',chrono_splits[-2][1] - chrono_splits[-1][1])
            if (chrono_splits[-2][1] - chrono_splits[-1][1]) < eps_:
                chrono_splits[-2][2] = None
                print('chrono before break ----->',chrono_splits,eps_)
                plt.show()
                break

        if best_t is not None: 
            chrono_splits[-2][2] = best_t
                
        # show line if line is neccessary
        fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize = (12, 3))
        plt.yticks([])
        ax1.imshow(image,cmap='gray')
        for i in range(len(done_splits)-1):
            if i != 0:
                ax1.axhline(done_splits[i][0],color='olive')
        #print('done with risk',done_splits)

        # calculate best split for each area
        for i in range(len(done_splits)-1):
            area = image[done_splits[i][0]:done_splits[i+1][0]]

            # calculate best split for this area
            if yet_splits[i] is None:
                best = best_split_general(area, criterion, ax3, y_start=done_splits[i][0], rate=rate, uni_axis='y', no_split=done_splits[i][1])
                if best is None: # image is not in the right form
                    return
                axis,t,emp_risk_split = best
                yet_splits[i] = (int(t)+done_splits[i][0],float(emp_risk_split),done_splits[i][1]-float(emp_risk_split))#,done_splits[i][1])

        ax3.scatter([s[0] for s in yet_splits],[s[2] for s in yet_splits],marker='x',color='red')
        
        # choose best split among areas
        #print('calculated yet',yet_splits)
        best_ind = np.argmax([t[2] for t in yet_splits])
        best_t   = yet_splits[best_ind][0]

        # insert new split in order
        ind = np.searchsorted([s[0] for s in done_splits],best_t)
        done_splits.insert(ind,(best_t,None)) 
        if ind > 0:
            done_splits[ind-1] = (done_splits[ind-1][0],None)
        
        print('---> new split at',best_t)
        ax1.axhline(best_t,color='red')
        ax3.axvline(best_t,color='red')
        for t,_,_ in yet_splits:
            ax1.axhline(t,color='red',linestyle='dashed')
        #    ax3.axvline(t,color='red',linestyle='dashed')
        #plt.show()

        # reset for new areas
        yet_splits[best_ind] = None
        yet_splits.insert(best_ind+1,None)
        print('chrono final',chrono_splits)

    return [('y',0),('y',len(image))]+[('y',s[2]) for s in chrono_splits if s[2] is not None and s[2] != 'None']

def cart_regression_bi(image,criterion,max_depth=10,eps_=8,rate=1,max_x=None,max_y=None):
    '''
    '''
    if image.size == 0 or max_depth <= 0:
        return []

    # color to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    if max_y is not None:
        y_splits = cart_regression_uni(image,criterion,max_y,eps_,rate)
    else:
        y_splits = cart_regression_uni(image,criterion,max_depth,eps_,rate)
    print('====>',y_splits)
    if max_x is not None:
        x_splits = cart_regression_uni(cv2.transpose(image),criterion,max_x,eps_,rate)
    else:
        x_splits = cart_regression_uni(cv2.transpose(image),criterion,max_depth,eps_,rate)
    print('====>',x_splits)
    return y_splits + [('x',i[1]) for i in x_splits]

def cart_regression_bi_simul(image,criterion,max_depth=10,eps_=8,rate=1):
    ''' 
    Parameters:

    Returns:
        Splits images iteratively while considering both directions simultaneously.
    '''
    if image.size == 0 or max_depth <= 0:
        return []

    # color to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    # normalization
    #if (image>1).all():
    #    image = image/255.0    
    
    areas=[[0,0,len(image[0]),len(image),None,None]]
    final_areas = []
    chrono_splits=[]
    yet_splits=[None]
    best_t = None
    best_d = None
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize = (12, 3))
    plt.yticks([])
    ax1.imshow(image,cmap='gray')
    
    while (all([s is None for s in yet_splits]) or max_depth > 0):
        print('DEPTH =',max_depth)
        #print('done',areas,'yet',yet_splits,'chrono',chrono_splits)

        # calculate risk of each area
        for i in range(len(areas)):
            if areas[i][4] is None:    
                x_s = areas[i][0]
                y_s = areas[i][1]
                x_e = areas[i][2]
                y_e = areas[i][3]
                area = image[y_s:y_e,x_s:x_e]
                if area.size == 0:
                    continue
                areas[i][4] = criterion(area,np.ones_like(area)+np.mean(area)-1)*area.size
                #areas[i][5] = criterion(area,np.ones_like(area)+np.mean(area)-1)
        
        chrono_splits.append([len(areas)-1,np.sum([a[4] for a in areas if a[4] is not None])/1e6,None,None])
        ax2.scatter(np.arange(len(chrono_splits)),[s[1] for s in chrono_splits])
        plt.show()
        if len(chrono_splits) > 1:
            #print('gain in empirical risk with splitting is',chrono_splits[-2][1] - chrono_splits[-1][1])
            print('diff = ',chrono_splits[-2][1] - chrono_splits[-1][1])
            if chrono_splits[-2][1] - chrono_splits[-1][1] < eps_:
                chrono_splits[-2][2] = None
                chrono_splits[-2][3] = None
                plt.show()
                break
        final_areas = [a[:4] for a in areas]

        if best_t is not None: 
            chrono_splits[-2][2] = best_t

        if best_d is not None:
            chrono_splits[-2][3] = best_d

        # show areas
        fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize = (12, 3))
        plt.yticks([])
        ax1.imshow(image,cmap='gray')
        for x1,y1,x2,y2,_,_ in areas:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='olive', fill=False)
            ax1.add_patch(rect)

        # calculate best split for each area
        new = set()
        for i in range(len(areas)):
            #others = np.sum([areas[j][4] for j in range(len(areas)) if j != i])
            area = image[areas[i][1]:areas[i][3],areas[i][0]:areas[i][2]]
            if area.size == 0:
                continue
            # calculate best split for this area
            if yet_splits[i] is None:
                new.add(i)
                best = best_split_general(area, criterion, ax3, x_start=areas[i][0], y_start=areas[i][1], rate=rate, no_split=areas[i][4])#, uni_axis='y')
                if best is None: # image is not in the right form
                    return
                axis,t,emp_risk_split = best
                #np.insert(yet_splits,i,[[t,emp_risk_split]])
                #no_split = criterion(area,np.ones_like(area)+np.mean(area)-1)
                if axis == 'x':
                    yet_splits[i] = (int(t)+areas[i][0],axis,float(emp_risk_split),areas[i][4]-float(emp_risk_split))        #,risk_general+float(emp_risk_split)
                else:
                    yet_splits[i] = (int(t)+areas[i][1],axis,float(emp_risk_split),areas[i][4]-float(emp_risk_split))        #,risk_general+float(emp_risk_split)
        
        #print('calculated splits',yet_splits)
        ax3.scatter([s[0] for s in yet_splits],[s[3] for s in yet_splits],marker='x',color='olive')
        for i in new:
            ax3.scatter(yet_splits[i][0],yet_splits[i][3],marker='x',color='red')
        # choose best split among areas
        best_ind = np.argmax([t[3] for t in yet_splits if t is not None])  #2
        best_t   = yet_splits[best_ind][0]
        best_d   =yet_splits[best_ind][1]

        ax3.axvline(best_t,color='red')
        for i in range(len(yet_splits)):
            if yet_splits[i][1] == 'x':
                #print(yet_splits[i],areas[i][1],areas[i][1]/len(image),areas[i][3],areas[i][3]/len(image))
                ax1.axvline(yet_splits[i][0],linestyle='dashed',color='red',ymax=1-areas[i][1]/len(image),ymin=1-areas[i][3]/len(image))
            else:
                #print(yet_splits[i],areas[i][0],areas[i][0]/len(image[0]),areas[i][2],areas[i][2]/len(image[0]))
                ax1.axhline(yet_splits[i][0],linestyle='dashed',color='red',xmin=areas[i][0]/len(image[0]),xmax=areas[i][2]/len(image[0]))

        # insert new split in order
        if yet_splits[best_ind][1] == 'x':   
            ax1.axvline(best_t,color='red',ymax=1-areas[best_ind][1]/len(image),ymin=1-areas[best_ind][3]/len(image))         
            areas.insert(best_ind+1,[best_t,areas[best_ind][1],areas[best_ind][2],areas[best_ind][3],None,None]) 
            areas[best_ind][2] = best_t
            areas[best_ind][4] = None
            #areas[best_ind][5] = None
            #chrono_splits.append(yet_splits[best_ind])
            print('---> new split at',best_t)
        else:            
            ax1.axhline(best_t,color='red',xmin=areas[best_ind][0]/len(image[0]),xmax=areas[best_ind][2]/len(image[0]))
            areas.insert(best_ind+1,[areas[best_ind][0],best_t,areas[best_ind][2],areas[best_ind][3],None,None]) 
            areas[best_ind][3] = best_t
            areas[best_ind][4] = None
            #areas[best_ind][5] = None
            #chrono_splits.append(yet_splits[best_ind])
            print('---> new split at',best_t)

        # reset for new areas
        yet_splits[best_ind] = None
        yet_splits.insert(best_ind+1,None)
        
        max_depth -= 1

    return final_areas
    #return [(s[3],s[2]) for s in chrono_splits if s[2] is not None and s[2] != 'None']           
