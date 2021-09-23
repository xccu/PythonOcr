# 需要安装：![](img/invincible0.jpg)
#   PaddleOCR       2.3.0.1
#   paddlepaddle    2.1.2
#   shapely         1.7.1
#   pyclipper       1.3.0

import os
os.environ['HUB_HOME'] = "./modules"
import paddlehub as hub

# PaddleHub一键OCR中文识别（超轻量8.1M模型，火爆
# https://aistudio.baidu.com/aistudio/projectdetail/507159?channelType=0&channel=0

# 加载移动端预训练模型
ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
# 服务端可以加载大模型，效果更好
#ocr = hub.Module(name="chinese_ocr_db_crnn_server")

import cv2
# 读取测试文件夹test.txt中的照片路径
# test_img_path = ['20210522205411.png']
# test_img_path = [r'D:\_00_workspaces\Machine_Learning\00_generate_captcha\20210522205411.png']
test_img_path = [r'./../PythonOcr/img/list1.jpg']
np_images =[cv2.imread(image_path) for image_path in test_img_path]

results = ocr.recognize_text(
                    images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                    use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                    # use_gpu=True,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                    output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                    visualization=True,       # 是否将识别结果保存为图片文件；
                    box_thresh=0.5,           # 检测文本框置信度的阈值；
                    text_thresh=0.5)          # 识别中文文本置信度的阈值；

for result in results:
    data = result['data']
    save_path = result['save_path']
    for infomation in data:
        print('text: ', infomation['text'],
              '\tconfidence: ', infomation['confidence'],
              '\ttext_box_position: ', infomation['text_box_position'])
