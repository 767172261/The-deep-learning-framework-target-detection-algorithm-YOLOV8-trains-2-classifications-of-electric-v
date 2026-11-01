深度学习框架目标检测算法YOLOV8训练 电动车自行车检测数据集  2分类，电动车和自行车检测数据集的训练及应用。
@[toc]

**数据集描述：**

电动车自行车检测数据集2分类，电动车和自行车。YOLO的目标检测。 图片共851张。数据集按7：2：1分配。其中 训练集596张图片，测试集170张图片，验证集85张
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0f5b475303994fc89705c3e376e5982f.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5c75e4a54a5a4b6eae375de3172a260a.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e6e41e31dfd94179bbdda88fe6a1b569.png)
 **电动车和自行车目标检测数据集**  YOLOv8 的小型目标检测任务。将从环境搭建、数据准备、模型训练、推理、评估等角度，为你提供一套 **完整详细的训练流程和代码**，

---

## 🧰 一、环境准备

### 1. 安装 CUDA 驱动（可选，如使用 GPU）

确保系统已经安装了 NVIDIA 显卡驱动和 CUDA 工具包。你可以通过运行以下命令检查：

```bash
nvidia-smi
```

### 2. 安装 Anaconda 并创建虚拟环境

```bash
conda create -n bike_ebike python=3.8
conda activate bike_ebike
```

### 3. 安装必要依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib tqdm
```

> ✅ 根据你的 CUDA 版本选择 PyTorch 安装命令，如 `cu118`（CUDA 11.8）。

---

## 📁 二、数据集结构

你提供的数据集包含 851 张图像，按 7:2:1 分为：

- 训练集：596 张
- 验证集：85 张
- 测试集：170 张

YOLO 格式数据结构如下：

```
bike_ebike_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### `data.yaml` 内容如下：

```yaml
train: ./bike_ebike_dataset/images/train
val: ./bike_ebike_dataset/images/val
test: ./bike_ebike_dataset/images/test

nc: 2
names: ['bicycle', 'electric_bike']
```

---

## 🚀 三、使用 YOLOv8 训练模型

### 1. 加载预训练模型

```python
from ultralytics import YOLO

# 加载官方YOLOv8n预训练模型
model = YOLO('yolov8n.pt')
```

### 2. 开始训练

```python
def train_yolo():
    model = YOLO('yolov8n.yaml')  # 从配置文件初始化模型（可选）
    results = model.train(
        data='./bike_ebike_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='ebike_train',
        project='runs/detect',
        save=True,
        save_period=10,
        device=0 if torch.cuda.is_available() else None
    )

if __name__ == '__main__':
    train_yolo()
```

> 📌 注意：
> - `device=0` 表示使用 GPU（单卡）
> - `save_period=10` 表示每10个epoch保存一次权重
> - 模型结果会保存在 `runs/detect/ebike_train/` 目录下

---


### 4. 评估模型

训练完成后，可以使用以下代码进行模型评估，并生成 Precision-Recall 曲线：

```python
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

def evaluate_model():
    # 加载训练好的模型
    model = YOLO('runs/detect/ebike_train/weights/best.pt')

    # 进行评估
    metrics = model.val()

    # 获取Precision-Recall曲线数据
    pr_curve_data = metrics.box.pr

    # 提取Precision和Recall值
    recall = pr_curve_data[:, 1]
    precision = pr_curve_data[:, 0]

    # 绘制Precision-Recall曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'all classes {metrics.box.map50:.3f} mAP@0.5')
    
    # 如果需要绘制每个类别的Precision-Recall曲线
    for i in range(metrics.nc):
        class_recall = pr_curve_data[pr_curve_data[:, 2] == i, 1]
        class_precision = pr_curve_data[pr_curve_data[:, 2] == i, 0]
        plt.plot(class_recall, class_precision, label=f'class {i} {metrics.box.ap[i]:.3f}')

    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.show()

if __name__ == '__main__':
    evaluate_model()
```

### 5. 解释结果

 Precision-Recall 曲线图中可以看到：

- **所有类别** 的 mAP@0.5 为 0.906，表示模型的整体性能非常好。
- **自行车（bicycle）** 类别的 mAP@0.5 为 0.924，略高于电动车（motorcycle）类别。
- **电动车（motorcycle）** 类别的 mAP@0.5 为 0.888，也表现出色。




![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/609b66a01b16493f837da5b444c91746.png)

## 🔍 四、模型推理代码（图像检测）

```python
from ultralytics import YOLO
import cv2
import os

def detect_image(image_path, output_dir='runs/detect/results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = YOLO('runs/detect/ebike_train/weights/best.pt')  # 替换为你的模型路径
    results = model.predict(source=image_path, save=True, project=output_dir, name="")

    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, im)

    print(f"检测完成，结果保存在 {output_path}")

if __name__ == '__main__':
    detect_image('path/to/test/image.jpg')
```

---

## 📊 五、模型评估代码（验证集）

```python
from ultralytics import YOLO

def evaluate_yolo():
    model = YOLO('runs/detect/ebike_train/weights/best.pt')
    metrics = model.val()  # 自动使用验证集进行评估

    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

if __name__ == '__main__':
    evaluate_yolo()
```

输出示例：

```
mAP50: 0.906
mAP50-95: 0.782
```

---

## 📈 六、训练结果可视化

训练结束后，YOLOv8 会自动在 `runs/detect/ebike_train/` 生成以下文件：

- `weights/`: 包含 `best.pt` 和 `last.pt` 权重文件
- `results.csv`: 每轮训练的 loss 和 mAP 指标
- `plots/`: 包含训练过程中的 loss 曲线、PR 曲线、混淆矩阵等图表

使用 Excel 或 Python 加载 `results.csv` 进行进一步分析：

```python
import pandas as pd
df = pd.read_csv('runs/detect/ebike_train/results.csv')
print(df.tail())
```

---



## 📦 九、附：requirements.txt

```txt
torch
torchvision
ultralytics
opencv-python
numpy
matplotlib
tqdm
```

---


