import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 读取一张图片并显示出来 
img = cv2.imread("test.jpg")  # 读取图片

cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 在图片中插入文字（学号+姓名）
img_text = img.copy()  # 复制图片

# 用PIL来写中文
img_pil = Image.fromarray(cv2.cvtColor(img_text, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 36)  
draw.text((50, 50), "23120899 周梓俊", font=font, fill=(255, 0, 0))
img_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

cv2.imshow("with text", img_text)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 保存图片
cv2.imwrite("TextedImg.jpg", img_text)

# 读取本地视频并播放
cap = cv2.VideoCapture("Waymo.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频读完了就退出
    cv2.imshow("Waymo", frame)
    # 延时25毫秒，控制播放速度并刷新窗口。
    cv2.waitKey(25)
    
    # 点右上角的叉退出
    if cv2.getWindowProperty("Waymo", cv2.WND_PROP_VISIBLE) == 0:
        break

cap.release()
cv2.destroyAllWindows()
