from flask import Flask, jsonify, render_template
import cv2
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision.transforms.functional import to_pil_image
from rembg import remove
from collections import defaultdict

# 紀錄垃圾類別累積數量
category_counts = defaultdict(int)
history_log = defaultdict(list)  # 時間序列資料


# 建立反標準化的 transform（ImageNet mean/std）
unnormalize = transforms.Normalize(
    mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    std=[1/s for s in [0.229, 0.224, 0.225]]
)

# 儲存處理過的圖片
def save_tensor_as_image(tensor, filename):
    img = unnormalize(tensor.squeeze(0)).clamp(0, 1)  # 去 batch 維 + 限定 0~1
    img_pil = to_pil_image(img)
    os.makedirs('rotated_image', exist_ok=True)
    img_pil.save(os.path.join('rotated_image', filename))

def rotate_and_pad_dynamic(image, angle, background_color=(0, 0, 0), min_size=400):
    # Step 1: 旋轉並展開
    # image = remove(image)
    rotated = image.rotate(angle, expand=True, fillcolor=background_color)

    # Step 2: 動態計算正方形背景尺寸（取最大邊長，與 min_size 比較）
    side = max(min_size, rotated.width, rotated.height)
    
    # Step 3: 建立正方形背景並貼上圖片
    background = Image.new("RGB", (side, side), background_color)
    paste_x = (side - rotated.width) // 2
    paste_y = (side - rotated.height) // 2
    background.paste(rotated, (paste_x, paste_y))

    return background

app = Flask(__name__)

# 初始化相機
 # 0 是預設相機
os.makedirs('captured_images', exist_ok=True)
dir_name = 'captured_images'

@app.route('/capture', methods=['GET'])
def capture_image():
    cap = cv2.VideoCapture(0) 
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'photo_{timestamp}.jpg'
        cv2.imwrite(os.path.join(dir_name, filename), frame)

        # 設定測試圖片資料夾
        test_image_dir = 'captured_images/'

        # 模型輸入所需的轉換（依照你訓練時使用的 transform）
        transform = transforms.Compose([
            transforms.Lambda(lambda img: rotate_and_pad_dynamic(img, angle=random.randint(-180, 180))),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 載入你訓練好的模型
        testmodel = torch.load('waste_classification_model_resnet.pth', map_location=torch.device('cpu'))
        testmodel.eval()
        # 類別標籤（順序需與訓練時相符）
        class_names = ['aluminum_soda_cans', 'cardboard_boxes', 'paper_cups', 'plastic_water_bottles']

        # 預測每一張圖片
        for filename in os.listdir(test_image_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(test_image_dir, filename)
                image = Image.open(image_path).convert('RGB')
                image = remove(image)
                
                input_tensor = transform(image).unsqueeze(0)  # 增加 batch dimension
                save_tensor_as_image(input_tensor, filename)  # 儲存處理過的圖片
                
                with torch.no_grad():
                    output = testmodel(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    print(probabilities)
                    predicted_class = class_names[output.argmax(1).item()]
                class_num = 0
                if predicted_class.startswith('aluminum_') or predicted_class.startswith('steel'):
                    print(f"{filename} → 預測為: 鐵鋁罐")
                    class_num = 1
                elif predicted_class.startswith('cardboard_'):
                    print(f"{filename} → 預測為: 紙箱/紙板")
                    class_num = 2
                elif predicted_class.startswith('plastic'):
                    print(f"{filename} → 預測為: 寶特瓶")
                    class_num = 3
                elif predicted_class.startswith('paper_'):
                    print(f"{filename} → 預測為: 紙杯")
                    class_num = 4
                os.remove(image_path)
        # 在預測完成後
        category_counts[class_num] += 1
        history_log[class_num].append(category_counts[class_num])
        
        return str(class_num), 200
    else:
        return "Failed to capture", 500

@app.route('/status')
def get_status():
    return jsonify({
        "counts": category_counts,
        "history": history_log
    })

@app.route('/')
def dashboard():
    return render_template("web.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
