import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取图片，转换为HSV空间
img = cv2.imread("home_color.png")
h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 分离RGB通道和HSV通道
b, g, r = cv2.split(img)
h_ch, s_ch, v_ch = cv2.split(hsv)

# 显示并保存RGB三个通道
cv2.imshow("R channel", r)
cv2.imshow("G channel", g)
cv2.imshow("B channel", b)
cv2.imwrite("R_channel.png", r)
cv2.imwrite("G_channel.png", g)
cv2.imwrite("B_channel.png", b)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 显示并保存HSV三个通道
cv2.imshow("H channel", h_ch)
cv2.imshow("S channel", s_ch)
cv2.imshow("V channel", v_ch)
cv2.imwrite("H_channel.png", h_ch)
cv2.imwrite("S_channel.png", s_ch)
cv2.imwrite("V_channel.png", v_ch)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 对RGB三个通道分别画出三维图（plot_surface）
# 创建网格坐标
rows = np.arange(0, img.shape[0])
cols = np.arange(0, img.shape[1])
X, Y = np.meshgrid(cols, rows)

step = max(1, max(img.shape[0], img.shape[1]) // 100)
X_s = X[::step, ::step]
Y_s = Y[::step, ::step]

fig = plt.figure(figsize=(15, 5))

# R通道三维图
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_s, Y_s, r[::step, ::step], cmap='Reds')
ax1.set_title('R Channel')

# G通道三维图
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X_s, Y_s, g[::step, ::step], cmap='Greens')
ax2.set_title('G Channel')

# B通道三维图
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X_s, Y_s, b[::step, ::step], cmap='Blues')
ax3.set_title('B Channel')

plt.tight_layout()
plt.savefig("rgb_3d_surface.png")
plt.show()



home_color = cv2.imread("home_color.png")

# 灰度直方图，拼接原灰度图与结果图
home_gray = cv2.cvtColor(home_color, cv2.COLOR_BGR2GRAY)

fig2, axes = plt.subplots(2, 2, figsize=(10, 8))

# 灰度图和灰度直方图
axes[0, 0].imshow(home_gray, cmap='gray')
axes[0, 0].set_title('Gray Image')
axes[0, 0].axis('off')

gray_hist = cv2.calcHist([home_gray], [0], None, [256], [0, 256])
axes[0, 1].plot(gray_hist, color='black')
axes[0, 1].set_title('Gray Histogram')
axes[0, 1].set_xlim([0, 256])

# 彩色直方图，拼接原彩色图与结果图，和上面放在同一个窗口
home_rgb = cv2.cvtColor(home_color, cv2.COLOR_BGR2RGB)
axes[1, 0].imshow(home_rgb)
axes[1, 0].set_title('Color Image')
axes[1, 0].axis('off')

colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([home_color], [i], None, [256], [0, 256])
    axes[1, 1].plot(hist, color=col)
axes[1, 1].set_title('Color Histogram')
axes[1, 1].set_xlim([0, 256])

plt.tight_layout()
plt.savefig("histogram_gray_color.png")
plt.show()

# ROI直方图，ROI区域 x:50-100, y:100-200
# 创建mask
mask = np.zeros(home_gray.shape[:2], dtype=np.uint8)
mask[100:200, 50:100] = 255  # y:100-200, x:50-100

# 用mask提取ROI区域
roi_img = cv2.bitwise_and(home_color, home_color, mask=mask)
roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)

# ROI直方图
roi_hist = cv2.calcHist([home_gray], [0], mask, [256], [0, 256])

fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))

# 原图
axes3[0].imshow(home_rgb)
axes3[0].set_title('Original')
axes3[0].axis('off')

# mask图
axes3[1].imshow(mask, cmap='gray')
axes3[1].set_title('Mask')
axes3[1].axis('off')

# ROI提取后的图
axes3[2].imshow(roi_rgb)
axes3[2].set_title('ROI Result')
axes3[2].axis('off')

# ROI直方图
axes3[3].plot(roi_hist, color='black')
axes3[3].set_title('ROI Histogram')
axes3[3].set_xlim([0, 256])

plt.tight_layout()
plt.savefig("roi_histogram.png")
plt.show()


img3 = cv2.imread("home_color.png")
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

equalized = cv2.equalizeHist(gray3)

fig4, axes4 = plt.subplots(2, 2, figsize=(10, 8))

# 原灰度图
axes4[0, 0].imshow(gray3, cmap='gray')
axes4[0, 0].set_title('Original Gray')
axes4[0, 0].axis('off')

# 原图直方图
hist_before = cv2.calcHist([gray3], [0], None, [256], [0, 256])
axes4[0, 1].plot(hist_before, color='black')
axes4[0, 1].set_title('Before Equalization')
axes4[0, 1].set_xlim([0, 256])

# 均衡化后的图
axes4[1, 0].imshow(equalized, cmap='gray')
axes4[1, 0].set_title('Equalized Gray')
axes4[1, 0].axis('off')

# 均衡化后的直方图
hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])
axes4[1, 1].plot(hist_after, color='black')
axes4[1, 1].set_title('After Equalization')
axes4[1, 1].set_xlim([0, 256])

plt.tight_layout()
plt.savefig("histogram_equalization.png")
plt.show()
