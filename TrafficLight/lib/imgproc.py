from turtle import color

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 展示图片函数
def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')  # 关闭坐标轴
    plt.show()

# 转换为jpeg格式，用于控件显示
def to_jpeg_bytes(frame, is_mask=False):
    if is_mask:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

def process_frame(frame, color_range, lowest=30, highest=100):
    # 参数整理
    color1_low = np.array(color_range[0])
    color1_high = np.array(color_range[1])
    color2_low = np.array(color_range[2])
    color2_high = np.array(color_range[3])
    # 图像处理，生成掩膜
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, color1_low, color1_high)
    mask2 = cv2.inRange(hsv, color2_low, color2_high)
    mask = cv2.bitwise_or(mask1, mask2)
    # 降噪和膨胀
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw = frame.copy()
    # for c in contours:
    #     area = cv2.contourArea(c)
    #     if lowest < area < 100:
    #         x, y, w, h = cv2.boundingRect(c)
    #         cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #         cv2.circle(draw, center, radius, (0, 255, 0), -1)

    # return draw, mask
    # 寻找轮廓，计算面积，找出最大
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0; max_contour = None;draw=frame.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_contour = c
    if max_contour is not None:
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(draw, center, radius, (0, 255, 0), -1)
    return draw, mask