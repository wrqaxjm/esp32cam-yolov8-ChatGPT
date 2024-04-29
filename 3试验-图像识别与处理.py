import os
import time
from collections import defaultdict
from threading import Thread
import cv2
import requests
import shutil
from ultralytics import YOLO
import json

# 定义ESP32-CAM的IP地址
esp32_cam_ip = '192.168.139.157'
# 定义ESP32-CAM Web服务器的路由
image_url = f'http://{esp32_cam_ip}/cam.jpg'  # 获取静态图像的路由
# 加载YOLOv8模型
model = YOLO("E:/yolo-project/yolov8/ultralytics-main/yolov8x.pt")
# 设置输出文件夹路径
output_folder = 'E:/yolo-project/yolov8/ultralytics-main/实验图片/实验2.1'
image_folder = 'E:/yolo-project/yolov8/ultralytics-main/实验图片/实验2.2'
outputfolder = 'E:/yolo-project/yolov8/ultralytics-main/实验图片/实验2.2处理后的图片'
# 确保输出文件夹存在
if not os.path.exists(output_folder):
    try:
        os.makedirs(output_folder)
        print(f"文件夹 {output_folder} 创建成功")
    except OSError as e:
        print(f"创建文件夹 {output_folder} 失败，原因：{e.strerror}")

# 定义不同类别对应的颜色（示例颜色）
class_colors = {
    "person": (255, 0, 0),   # 红色
    "tv": (0, 255, 0),       # 绿色
    "laptop": (0, 0, 255)    # 蓝色
}

# 控制变量
flag = True

# 图像获取的线程函数


def image_collect():
    global flag
    while flag:
        # 获取图像
        response = requests.get(image_url)
        if response.status_code == 200:
            # 解码图像数据
            image_data = response.content
            # 保存图像到临时文件夹中
            temp_image_path = os.path.join(output_folder, f"temp_image_{int(time.time())}.jpg")  # 使用时间戳命名图片
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
            print("保存临时图片成功:", temp_image_path)
            # 检查临时文件夹中图片的数量
            temp_image_count = len(os.listdir(output_folder))
            if temp_image_count >= 200:
                # 删除最先收集的100张图片
                temp_images = sorted(os.listdir(output_folder), key=lambda x: os.path.getctime(os.path.join(output_folder, x)))
                for i in range(100):
                    os.remove(os.path.join(output_folder, temp_images[i]))
                print("删除最先收集的100张图片成功。")
            # 检查指定文件夹中图片的数量
            image_count = len([name for name in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, name))])
            if image_count < 20:
                # 复制临时图片到指定文件夹，直到达到20张为止
                new_image_path = os.path.join(image_folder, f"image_{image_count + 1}.jpg")
                shutil.copy(temp_image_path, new_image_path)
                print("复制图片成功:", new_image_path)
            else:
                print("指定文件夹中图片数量已达到20张，继续获取图像...")
        else:
            print("获取图像失败，请检查网络连接或图像URL")
        time.sleep(0.5)  # 每秒获取一次图像

# 图像处理的线程函数
# 定义全局变量


zhixingdu = None
zhonglei = None
symbol = 0


def image_processing():
    global flag, zhixingdu, zhonglei,  symbol  # 在函数内部引用全局变量
    while flag:
        # 检查图片数量是否达到处理要求
        image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) >= 20:
            # 开始图像处理
            # 获取图像文件夹中的所有图像文件，并按文件修改时间排序
            image_files.sort(key=os.path.getmtime)
            # 逐个处理图像
            for i, image_path in enumerate(image_files, start=1):
                frame = cv2.imread(image_path)
                # 调整图像大小为1280x960
                resized_frame = cv2.resize(frame, (1280, 960))
                # 进行目标检测
                results_list = model(resized_frame)
                for results in results_list:
                    if results.boxes is not None:
                        xyxy_boxes = results.boxes.xyxy
                        conf_scores = results.boxes.conf
                        cls_ids = results.boxes.cls
                        for box, conf, cls_id in zip(xyxy_boxes, conf_scores, cls_ids):
                            x1, y1, x2, y2 = map(int, box)
                            cls_id = int(cls_id)
                            label = model.names[cls_id]
                            confidence = f"{conf:.2f}"
                            # 将置信度和类别信息赋值给全局变量
                            zhixingdu = confidence
                            zhonglei = label
                            print(zhixingdu)
                            print(zhonglei)
                            rectangle_color = class_colors.get(label, (255, 255, 255))  # 如果没有定义的类别颜色，默认为白色
                            label_color = (0, 0, 255)
                            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), rectangle_color, 2)
                            cv2.putText(resized_frame, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
                # 保存处理后的图像
                processed_image_path = os.path.join(outputfolder, f"processed_image_{i}.jpg")
                cv2.imwrite(processed_image_path, resized_frame)
                print("保存处理后的图片:", processed_image_path)
            # 清空图像文件夹
            for file in os.listdir(image_folder):
                os.remove(os.path.join(image_folder, file))
            print("清空图像文件夹成功。")
            symbol = 1
            send_messages()
        else:
            print("等待更多图像以开始处理...")
            time.sleep(5)


def get_ChatMindAi_answer(modified_keyword):
    ChatMindAiUrl = "https://api.chatanywhere.com.cn/v1/chat/completions"
    ChatMindAiApiKey = "sk-tetT9sM4MSA8a3LGJcZGxfuyN4c0fAfrcVm9uwqAJA2Yt3bq"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + ChatMindAiApiKey
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": modified_keyword
            }
        ]
    }
    response = requests.post(ChatMindAiUrl, json=data, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        answer = response_json["choices"][0]["message"]["content"]
        return answer

    else:
        print("HTTP POST request failed, error code:", response.status_code)
        return "<error>"


def send_answer_to_esp32(answer):
    esp32_url = 'http://192.168.139.25/detection_results'  # 替换成ESP32的IP地址
    # 将回答文本数据编码为 UTF-8
    # 将回答文本数据编码为 UTF-8
    encoded_answer = answer.encode('utf-8')

    response = requests.post(esp32_url, data=encoded_answer)  # 发送编码后的回答文本数据

    if response.status_code == 200:
        print("Answer sent to ESP32 successfully")
    else:
        print("Failed to send answer to ESP32")

def send_messages():
    global flag, zhonglei
    modified_keyword = f"ChatMindAi，我通过摄像头和代码采集了一张现实世界的图片，并识别其中主要物体。现在我来告诉你，你面前的场景或物品是{zhonglei}，你需要做出回应，假装你真的在观察世界."
    answer = get_ChatMindAi_answer(modified_keyword)

    if answer is not None:  # 检查回答是否为空
        print("ChatMindAi回答：", answer)
        send_answer_to_esp32(answer)  # 将回答传输给esp32设备端
        print("已发送")

    else:
        print("ChatMindAi没有给出有效的回答。")



# 启动图像获取和处理的线程

collect_thread = Thread(target=image_collect)
process_thread = Thread(target=image_processing)


collect_thread.start()
process_thread.start()


# 等待所有线程结束
collect_thread.join()
process_thread.join()


print("程序结束。")
