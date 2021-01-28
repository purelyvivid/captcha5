from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN

def load_img(fname):
    img = Image.open(fname).convert('RGB')
    return img

def get_bin_count(golden_fnames):
    key = get_key()
    bin_count = [0]*len(key)
    for fname in golden_fnames:
        try:
            label = fname.split("/")[-1][0]
            bin_count[key.index(label)] += 1
        except:
            continue
    return bin_count, key

def load_golden(golden_fnames):
    key = get_key()
    g_imgarrs = []
    g_char_labels = []
    for g_fname in golden_fnames:
        g_img = Image.open(g_fname)
        g_imgarr = np.array(g_img).astype(float)
        #g_imgarr /= g_imgarr.max()
        g_char_labels.append(g_fname.split('/')[-1].split('.')[0][0])
        g_imgarrs.append(g_imgarr)
    g_imgarrs = np.stack(g_imgarrs, 0)
    
    g_int_labels = [key.index(l) for l in g_char_labels]
    #g_onehot_labels = np.zeros((g_imgarrs.shape[0], len(key)))
    #g_onehot_labels[range(g_imgarrs.shape[0]),g_int_labels] = 1
    
    return g_imgarrs, g_int_labels, key

def get_background(fnames):
    imgarrs_27_77, imgarrs_25_70 = [], []
    for fn in fnames:
        img = load_img(fn)
        imgarr = np.array(img)
        if img.size == (77, 27):
            imgarrs_27_77.append(imgarr)
        elif img.size == (70, 25):
            imgarrs_25_70.append(imgarr)
        else:
            print(fn, "shape error!", img.size)
            
    imgarrs_27_77_m, imgarrs_25_70_m = None, None

    if len(imgarrs_27_77)>0:
        imgarrs_27_77 = np.stack(imgarrs_27_77,0)
        imgarrs_27_77_m = np.mean(imgarrs_27_77, 0)
        print("27_77:", imgarrs_27_77.shape)
    if len(imgarrs_25_70)>0:
        imgarrs_25_70 = np.stack(imgarrs_25_70,0)
        imgarrs_25_70_m = np.mean(imgarrs_25_70, 0)
        print("25_70:", imgarrs_25_70.shape)
    return {(77, 27): imgarrs_27_77_m, (70, 25): imgarrs_25_70_m }

def img_resize(imgarr, img_size=(70,25)):
    img = Image.fromarray( imgarr.astype("uint8") )
    if img.size!=img_size:
        img = img.resize(img_size)
    imgarr = np.array(img) 
    return imgarr

def remove_background_n_resize(img, background_d, img_size=(70,25)):
    
    imgarr = np.array(img).astype("float64")   

    # remove bg
    if img.size in background_d:
        #print(imgarr.shape, background_d[img.size].shape)
        imgarr -= background_d[img.size]
        imgarr[imgarr<0] = 0
    else:
        return None
    
    while (imgarr[:,-1,2]>1).all():
        imgarr = imgarr[:,:-1,:]
    
    #resize
    imgarr = img_resize(imgarr, img_size=img_size)
    
    return imgarr/255

def smooth_img(imgarr,  kernel_size=3):
    # 2D Convolution ( Image Filtering )
    kernel = np.ones((kernel_size,kernel_size))
    kernel = kernel/np.sum(kernel)
    imgarr = cv2.filter2D(imgarr,-1,kernel)
    return imgarr

def shrink_scale(imgarr):
    imgarr = np.log(imgarr+1)
    imgarr/=imgarr.max() 
    return imgarr


def img_preprocess(fname, background_d, show_main=False, show_detail=False, 
                     output_size=20, pad_size=4,
                     n_char=5):#img.size=(25,70,4)
    
    img = load_img(fname) 
    
    if show_main or show_detail:
        print("原圖")     
        plt.imshow(img)
        plt.show() 
        
    if (not show_main) and show_detail:    
        print("區分前景和背景(去除背景)") 
        
    imgarr = remove_background_n_resize(img, background_d)
    img_size = imgarr.shape[:2]
    n_node = np.product(img_size)
    n_chnl = imgarr.shape[-1]
    
    if (not show_main) and show_detail:    
        plt.matshow(imgarr)
        plt.show()
        #_ = plt.hist(imgarr[...,0].flatten());plt.show()
        #_ = plt.hist(imgarr[...,1].flatten());plt.show()
        #_ = plt.hist(imgarr[...,2].flatten());plt.show()    
        print("將前景圖片顏色做調整") 
         
    imgarr = shrink_scale(imgarr)
    imgarr = shrink_scale(imgarr)
    imgarr = shrink_scale(imgarr)
    
    if (not show_main) and show_detail:    
        plt.matshow(imgarr)
        plt.show()
        #_ = plt.hist(imgarr[...,0].flatten());plt.show()
        #_ = plt.hist(imgarr[...,1].flatten());plt.show()
        #_ = plt.hist(imgarr[...,2].flatten());plt.show()    
        
     
    new_img = np.zeros_like(imgarr)
    take_pos = np.zeros(img_size).astype(bool)#前景的位置
    for col in range(img_size[1]):
        col_val = imgarr[:,col,:]#選定一欄
        selected_chnl = np.argsort(col_val.sum(0),axis=-1) #找該欄總和最大次大最小的channels
        col_max_val = imgarr[:,col,selected_chnl[-1]] #數值最大的channels
        col_val_ = col_max_val
        if np.max(col_val)<0.1:
            col_val_bool = np.zeros_like(col_val_).astype(bool)
        else:    
            col_val_bool = (col_val_>0.3)
        
        new_img[:,col,selected_chnl[-1]] = col_val_bool.astype(float)
        take_pos[:,col]= col_val_bool 
    
    if show_main or show_detail: 
        print("取出前景") 
        plt.matshow(new_img)
        plt.show()

    
    return new_img,take_pos

def correct_color_for_vertical_line(imgarr):#(H,W,3)
    #解決去背後產生 與眾不同垂直線的問題
    h, w = imgarr.shape[:2]
    imgarr_ = imgarr.copy()
    for r in range(h):
        for c in range(w):
            if c<2 or c>=w-2 or r<1 or r>=h-1:
                #外圍強制歸零
                imgarr_[r,c] = np.array([0,0,0])
                continue
            else:

                this = imgarr[r,c]    #(3,)
                left = imgarr[r,c-1] 
                right = imgarr[r,c+1]
                up = imgarr[r-1,c] 
                down = imgarr[r+1,c]
                
                u_is_zero = sum(up)==0
                d_is_zero = sum(down)==0
                l_is_zero = sum(left)==0
                r_is_zero = sum(right)==0
                same_l_r = (right==left).all()
                same_w_l = (this==left).all()
                same_w_r = (this==right).all()  
                
                if sum(this)==0:
                    if same_l_r and (not l_is_zero):
                        imgarr_[r,c] = left
                    else:  
                        #值為零 不處理
                        continue
                        
                elif l_is_zero and r_is_zero and u_is_zero and d_is_zero:
                    #周圍都為零 強制歸零
                    imgarr_[r,c] = np.array([0,0,0]) 
                        
                elif same_l_r:
                    #左邊與右邊鄰居相同   
                    if (not l_is_zero) :
                        imgarr_[r,c] = left
                    elif (not u_is_zero) :
                        imgarr_[r,c] = up
                    elif (not d_is_zero) :
                        imgarr_[r,c] = down
                    else:
                        continue
                        
                else:
                    
                    #左邊與右邊鄰居不同
                    if l_is_zero:
                        imgarr_[r,c] = right
                    elif r_is_zero:
                        imgarr_[r,c] = left
                    else:    
                        #左邊與右邊鄰居不同, 且左邊與右邊沒有人是零
                        left2 = imgarr[r,c-2] 
                        right2 = imgarr[r,c+2]

                        l2_is_zero = sum(left2)==0
                        r2_is_zero = sum(right2)==0
                        same_l2_r2 = (right2==left2).all()
                        same_l_l2 = (left==left2).all()
                        same_r_r2 = (right==right2).all()
                        same_w_l2 = (this==left2).all()
                        same_w_r2 = (this==right2).all()
                        
                        if (same_l2_r2 and (not l2_is_zero)) and ( same_l_l2 or same_r_r2):
                            imgarr_[r,c] = left2
                        elif same_w_l and same_l_l2:
                            imgarr_[r,c] = left
                        elif same_w_r and same_r_r2:
                            imgarr_[r,c] = right
                        else:
                            continue
                            


    return imgarr_
        
def img_segmentation(fname, background_d, show_main=False, show_detail=False, 
                     output_size=20, pad_size=4,
                     n_char=5):#img.size=(25,70,4)
    
    new_img,take_pos = img_preprocess(fname, background_d, show_main=show_main, show_detail=show_detail)
    
    # 去除垂直線雜點
    new_img = correct_color_for_vertical_line(new_img)

    img_size = new_img.shape[:2]
    n_node = np.product(img_size)
    n_chnl = new_img.shape[-1]
    
    if show_main or show_detail:   
        print("去除垂直線雜點") 
        plt.matshow(new_img)
        plt.show() 
        
    # 對前景做區分五組字元(此時考慮位置進去), 並依照順序排好
    coord2pos_array = np.array(range(n_node)).reshape(img_size)
    pos2coord_array = np.array([tuple([ int(coord_ ) for coord_ in  np.where(coord2pos_array==pos) ]) for pos in range(n_node)] )
    norm_pos2coord_array = (pos2coord_array-pos2coord_array.mean())/pos2coord_array.std() 
    
    new_img_flat = new_img.reshape((n_node,-1))
    img_flat_w_coord = np.concatenate([new_img_flat, norm_pos2coord_array], 1)
    #print(img_flat_w_coord.shape)
    num_clusters = n_char
    img_flat_w_coord_rm_bg = img_flat_w_coord[take_pos.flatten()]
    clustering = KMeans(n_clusters=num_clusters).fit( img_flat_w_coord_rm_bg )
    #print(np.unique(clustering.labels_))
    
    #  五組字元依照順序排好, 並去除數字本體以外的雜點        
    labels = clustering.labels_
    coord_y_list = []
    for label in range(num_clusters):
        img_label_i_coord = img_flat_w_coord_rm_bg[labels==label,-2:]*pos2coord_array.std() 

        #去除數字本體以外的雜點
        clustering_noisy_cleaning = DBSCAN(eps=2**0.5,min_samples=1).fit(img_label_i_coord)
        i = stats.mode(clustering_noisy_cleaning.labels_)[0][0]#返回眾數
        digit_noise = labels[labels==label].copy()
        digit_noise[clustering_noisy_cleaning.labels_!=i] = -1
        labels[labels==label] = digit_noise

        coord_y_list.append(img_label_i_coord[:, -1].mean())
        
    sorted_label = list(np.argsort(coord_y_list))
    new_labels = [ sorted_label.index(l)+1 if l in sorted_label else 0 for l in labels ]
    
    label_im = np.zeros(n_node)
    label_im[take_pos.flatten()] = new_labels # 0:背景 ,1~5:第1~5個字元
    label_im = label_im.reshape(img_size)
    
    if (not show_main) and show_detail: 
        print("五組字元依照順序排好, 並去除數字本體以外的雜點") 
        plt.matshow(label_im)
        plt.show()    

        
    #個別取出五組字元,並各自調整大小成相同size
    img_i_list = []
    for i in range(num_clusters):
        img_i = (label_im==i+1).astype(float)
        c1,c2 = np.array(range(img_size[0])), np.array(range(img_size[1]))
        m1,m2 = np.mean(img_i, 1), np.mean(img_i, 0)
        #plt.plot(m1);plt.show()
        max1,max2,min1,min2 = np.max(c1[m1!=0]), np.max(c2[m2!=0]), np.min(c1[m1!=0]), np.min(c2[m2!=0])
        img_i = img_i[min1:max1+1,min2:max2+1]
        img_i = img_resize(img_i, img_size=(output_size,output_size))
        img_i = np.pad(img_i, pad_size,  mode='constant')
        img_i_list.append(img_i)
        
    imgarr = np.concatenate(img_i_list, 1)

    if show_main or show_detail:  
        print("個別取出五組字元,並各自調整大小成相同size")
        plt.matshow(imgarr)
        plt.show()  
    
    return img_i_list