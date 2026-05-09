import cv2
import numpy as np

img = cv2.imread("test.jpg")

h, w = img.shape[:2]  # 获取图片的高和宽

# (1) 平移：x轴平移100像素，y轴平移150像素
# 平移矩阵：[[1, 0, tx], [0, 1, ty]]
M_translate = np.float32([[1, 0, 100], [0, 1, 150]])
img_translate = cv2.warpAffine(img, M_translate, (w, h))

cv2.imshow("translate", img_translate)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("translate.jpg", img_translate)

# (2) 缩放
# 缩放到 1024*768
img_resize1 = cv2.resize(img, (1024, 768))
cv2.imshow("resize 1024x768", img_resize1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("resize_1024x768.jpg", img_resize1)

# 按比例缩小 60%
img_resize2 = cv2.resize(img, None, fx=0.6, fy=0.6)
cv2.imshow("resize 60%", img_resize2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("resize_60.jpg", img_resize2)

# (3) 翻转
# 水平翻转 flipCode=1
img_flip_h = cv2.flip(img, 1)
cv2.imshow("flip horizontal", img_flip_h)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("flip_horizontal.jpg", img_flip_h)

# 垂直翻转 flipCode=0
img_flip_v = cv2.flip(img, 0)
cv2.imshow("flip vertical", img_flip_v)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("flip_vertical.jpg", img_flip_v)

# 水平+垂直翻转 flipCode=-1
img_flip_hv = cv2.flip(img, -1)
cv2.imshow("flip h+v", img_flip_hv)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("flip_hv.jpg", img_flip_hv)

# (4) 旋转：给出旋转中心，旋转角度，对图片旋转
# 旋转中心取图片中心，旋转60度，缩放比例1.0
center = (w // 2, h // 2)
angle = 60
M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
img_rotate = cv2.warpAffine(img, M_rotate, (w, h))

cv2.imshow("rotate 60", img_rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("rotate_45.jpg", img_rotate)

# (5) 缩略：将图片缩小，放到原图的左上角
img_thumbnail = img.copy()
small = cv2.resize(img, (w // 4, h // 4))  # 缩小到原来的1/4
sh, sw = small.shape[:2]
# 把缩小后的图片放到原图的左上角
img_thumbnail[0:sh, 0:sw] = small

cv2.imshow("thumbnail", img_thumbnail)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("thumbnail.jpg", img_thumbnail)


# 读取一张新的图片，将其转换为灰度图片
img2 = cv2.imread("test.jpg")
if img2 is None:
    print("test2.jpg读取失败,检查路径")
    exit()
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 调整灰度图片为正方形（边长不小于500像素）
gh, gw = gray.shape[:2]
side = max(gh, gw, 500)  # 取最大边长，且不小于500
# 用 resize 把图调成正方形
gray_square = cv2.resize(gray, (side, side))

cv2.imshow("gray square", gray_square)
cv2.waitKey(0)
cv2.destroyAllWindows()

#用圆形掩膜对图片进行切片，并保存切片后的图像
# 创建一个全黑的掩膜，大小和正方形灰度图一样
mask = np.zeros((side, side), dtype=np.uint8)
# 在掩膜上画一个白色的圆，圆心在图片中心，半径为边长的一半
center2 = (side // 2, side // 2)
radius = side // 2
cv2.circle(mask, center2, radius, 255, -1)  # -1表示填充整个圆

# 用掩膜和灰度图做位运算，圆形以外的区域变成黑色
result = cv2.bitwise_and(gray_square, gray_square, mask=mask)

cv2.imshow("circle mask result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("circle_mask.jpg", result)
