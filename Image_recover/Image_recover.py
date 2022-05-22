# coding : utf-8
'''
@creat_time = 2022/4/21,18:30
@auther = MrCrimson
Emal : mrcrimson@163.com
'''
import cv2
import numpy as np
import random,time
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


# 对图像添加噪声
noise_rate = [0.4,0.8,0.6]
def make_noise(img):
    noise_img = np.zeros(img.shape,np.uint8)
    mask = np.zeros(img.shape, np.uint8)
    b,g,r = cv2.split(mask)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()  # 随机生成0-1之间的数字
            if rdn < 0.8:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                b[i][j] = 0
            elif rdn >= 0.8:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                b[i][j] = 1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()  # 随机生成0-1之间的数字
            if rdn < 0.4:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                g[i][j] = 0
            elif rdn >= 0.4:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                g[i][j] = 1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()  # 随机生成0-1之间的数字
            if rdn < 0.6:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                r[i][j] = 0
            elif rdn >= 0.6:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                r[i][j] = 1
    mask = cv2.merge([b,g,r])
    noise_img = mask * img
    cv2.imwrite('noise_img.png',noise_img)
    cv2.imwrite('mask.png',mask)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',noise_img)
    # cv2.waitKey(0)

    return [mask,noise_img]

def evaluate(recover,origin):
    evaluate = cv2.norm(origin-recover,cv2.NORM_L2)
    return evaluate

def img_recover(img):
    block_size=16
    resImg = np.copy(img)
    rows,cols,channel = img.shape
    count = 0
    noiseMask = np.array(img != 0, dtype='double')
    # print(noiseMask)

    with tqdm(total=(rows//block_size+1)*(cols//block_size+1), desc='test', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for x in range(rows//block_size+1):
            # if x<63:
            #     continue
            for y in range(cols//block_size+1):
                if (x+1)*block_size>rows:
                    goal_block_x=rows
                else:
                    goal_block_x=(x+1)*block_size
                if (y+1)*block_size>cols:
                    goal_block_y=cols
                else:
                    goal_block_y=(y+1)*block_size
                # print('goal block:{}-{},{}-{}'.format(x*16,goal_block_x, y*16,goal_block_y))
                if x<3:
                    row_l=0
                else:
                    row_l=block_size*(x-3)+6
                if rows<(x+3)*block_size+10:
                    row_h=rows
                else:
                    row_h=(x+3)*block_size+10
                # print(row_l, row_h)
                if y<3:
                    col_l=0
                else:
                    col_l=block_size*(y-3)+6
                if cols<(y+3)*block_size+10:
                    col_h=cols
                else:
                    col_h=(y+3)*block_size+10
                # print(col_l, col_h,'\n')
                for chan in range(channel):
                    '计算回归模型'
                    x_train = []
                    y_train = []
                    for i in range(row_l, row_h):
                        for j in range(col_l, col_h):
                            if noiseMask[i, j, chan] == 0.:
                                continue
                            # if i == row and j == col:
                            #     continue
                            x_train.append([i, j])
                            y_train.append([img[i, j, chan]])
                    if x_train == []: # 完全被抹除则直接放弃本区域
                        continue
                    Regression = LinearRegression()
                    Regression.fit(x_train, y_train)
                    '根据模型填值'
                    for xx in range(x*block_size,goal_block_x):
                        for yy in range(y*block_size,goal_block_y):
                            if noiseMask[xx, yy, chan] != 0.:
                                continue
                            resImg[xx, yy, chan] = Regression.predict([[xx, yy]])
                pbar.update(1)
    return resImg



if __name__ == '__main__':
    img = cv2.imread('test.jpg')
    mask,noise_img = make_noise(img)
    print(type(mask),type(noise_img))
    recover_img = img_recover(noise_img)
    cv2.imwrite('recover_img.png',recover_img)
    res1 = evaluate(recover_img,img)
    res2 = evaluate(noise_img,img)
    print(res1,res2)