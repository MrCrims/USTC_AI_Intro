# 图像恢复实验报告

#### 王冠杰 PB19081611

## 实验内容

 图像是一种非常长假你的信息载体，但是在图像的获取，传输，存储过程中可能由于各种原因使得图像受到噪声的影响。如何去除噪声的影响，恢复图像原本的信息是计算机视觉中的重要研究问题。本次实验要对图像添加高斯噪声，并对添加噪声的图像进行基于线性回归模型的去噪。

## 实验要求

生成受损图像

使用区域二元线性回归模型进行图像恢复

用两个图像的二范数来评估误差

## 实验过程

原图为

![test](D:\Working\AI_Intro\EXP2\report.assets\test.jpg)

## 图像添加噪声

制作一个噪声掩模，使其BGR三个通道分别满足0.8,0.4,0.6的噪声比例。

```python
def make_noise(img):
    noise_img = np.zeros(img.shape,np.uint8)
    mask = np.zeros(img.shape, np.uint8)
    b,g,r = cv2.split(mask)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()  # 随机生成0-1之间的数字
            if rdn < 0.8:  # 如果生成的随机数小于噪声比例则将该像素点添加0
                b[i][j] = 0
            elif rdn >= 0.8:  # 如果生成的随机数大于比例则将该像素点添加1
                b[i][j] = 1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()  
            if rdn < 0.4:  
                g[i][j] = 0
            elif rdn >= 0.4:  
                g[i][j] = 1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()  
            if rdn < 0.6: 
                r[i][j] = 0
            elif rdn >= 0.6:  
                r[i][j] = 1
    mask = cv2.merge([b,g,r])
    noise_img = mask * img
    cv2.imwrite('noise_img.png',noise_img)
    cv2.imwrite('mask.png',mask)

    return [mask,noise_img]
```

这里我们分别对三个通道填充{0,1}值。由于图片本身为1920*1080比较大，所以考虑随机生成0-1之间的随机数，根据相对于噪声比例的大小来确定该点像素值，这样做的依据是在足够多点的情况下均匀分布的数学期望就是噪声比例。添加噪声的方法就是生成了整个噪声掩模后与原图进行哈达玛积得到添加了噪声的图像。

添加了噪声之后的图像为

![noise_img](D:\Working\AI_Intro\EXP2\report.assets\noise_img.png)



## 线性回归模型实现图像恢复

基本思想就是对于图片分块分通道恢复，每一个小块分通道，每一个通道用一个对应的线性回归模型进行恢复。

恢复的图像

![recover_img](D:\Working\AI_Intro\EXP2\report.assets\recover_img.png)

## 实验结果及分析

加了噪声的图像与原图的二范数为242960.61974731626，而复原的图像与原图像的二范数为315413.4079030249。虽然实际上的二范数甚至增大了，但是从图片观感的角度讲，恢复的图像是成功的。对于恢复的图像比加噪的图像的二范数还大的原因我认为是由于利用了线性模型对每一个分块进行重新填充，导致实际上分块内几乎没有与原图一致的像素点而使得最终计算的二范数更大。