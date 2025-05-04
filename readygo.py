import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/pi/project_demo/lib')
# 导入麦克纳姆轮运动库
# from McLumk_Wheel_Sports import *

# 定义红色的HSV范围
# 红色在 HSV 中有两个不连续区间，故定义两个区域
# lower_red1 = np.array([0, 80, 80])
# upper_red1 = np.array([15, 255, 255])
# lower_red2 = np.array([165, 80, 80])
# upper_red2 = np.array([180, 255, 255])
lower_red1 = np.array([0, 80, 80])
upper_red1 = np.array([15, 255, 255])
lower_red2 = np.array([165, 80, 80])
upper_red2 = np.array([180, 255, 255])
# 定义绿色在 HSV 颜色空间的范围
lower_green = np.array([35, 70, 80])
upper_green = np.array([85, 255, 255])

def detect_green(image):
    if image is None:
        print("读取图像失败")
        exit()
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建掩膜
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 形态学操作去噪
    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遍历轮廓，筛选合适的绿色区域
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 1000:  # 根据实际情况调整面积阈值
            return cv2.boundingRect(contour)
    else:
        return ()
        

def detect_red2(hsv):
    if hsv is None:
        print("读取图像失败")
        exit()
    
    # 创建掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    cv2.imwrite('test.png',mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遍历轮廓，筛选合适的红灯区域
    for contour in contours:
        area = cv2.contourArea(contour)
        if 18 < area < 1000:  # 根据实际情况调整面积阈值
            return cv2.boundingRect(contour)
    else:
        return ()
   

def detect_red(image):
    if image is None:
        print("读取图像失败")
        exit()
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    cv2.imwrite('test.png',mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遍历轮廓，筛选合适的红灯区域
    for contour in contours:
        area = cv2.contourArea(contour)
        if 35 < area < 1000:  # 根据实际情况调整面积阈值
            return cv2.boundingRect(contour)
    else:
        return ()
        
# 红色检测函数
def detect_red_info(image, LOWEST, lower_red1=np.array([0, 70, 70]), upper_red1=np.array([15, 255, 255]), lower_red2=np.array([165, 70, 70]), upper_red2=np.array([180, 255, 255])):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask_origin = mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask_handled = mask
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if LOWEST < area < 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return mask_origin, mask_handled, draw

def is_red(hsv_tuple):
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    hsv_array = np.array(hsv_tuple)
    in_range1 = np.all((hsv_array >= lower_red1) & (hsv_array <= upper_red1))
    in_range2 = np.all((hsv_array >= lower_red2) & (hsv_array <= upper_red2))
    return in_range1 or in_range2

    

def is_green(hsv_tuple):

    hsv_array = np.array(hsv_tuple)
    # 检查 HSV 值是否在绿色范围内
    in_range = np.all((hsv_array >= lower_green) & (hsv_array <= upper_green))
    return in_range



def move(x, y, angle, move_speed=100, rotate_speed=100):
    try:
        # 处理左右移动
        if x != 0:
            if x < 0:
                move_left(move_speed)
            else:
                move_right(move_speed)
            time.sleep(abs(x) / move_speed)
            stop_robot()
        
        # 处理前后移动
        if y != 0:
            if y > 0:
                move_forward(move_speed)
            else:
                move_backward(move_speed)
            time.sleep(abs(y) / move_speed)
            stop_robot()
        
        # 处理原地旋转
        if angle != 0:
            if angle < 0:
                rotate_left(rotate_speed)
            else:
                rotate_right(rotate_speed)
            time.sleep(abs(angle) / rotate_speed)
            stop_robot()

    except KeyboardInterrupt:
        stop_robot()
        
    finally:
        # 清理资源
        try:
            del bot
        except NameError:
            pass