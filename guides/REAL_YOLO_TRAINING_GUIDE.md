# 🌊 How to Train a REAL YOLOv8 Flood Model
**For the "Hacker" / ML Engineer**

To make the model "legit" (actually detect floods instead of just cars/people), follow these 4 steps.

## 1. Find a Dataset (The Hard Part)
You need images of **Jakarta Floods**.
*   **Option A (Fast)**: Search [Roboflow Universe](https://universe.roboflow.com/search?q=flood) for "Flood" or "Water Level".
*   **Option B (Manual)**: Scrape Google Images/Twitter for "Banjir Jakarta" and download ~100-200 images.

## 2. Label the Data (The Boring Part)
You must teach the AI what "Flood" looks like.
1.  Go to [Roboflow.com](https://roboflow.com) (free).
2.  Upload your images.
3.  Draw boxes around:
    *   `flood_water` (The water itself)
    *   `submerged_car` (Cars in water)
    *   `flooded_house`
    *   `person_wading`
4.  **Export** the dataset. Select format: **YOLOv8**.

## 3. The Training Code (The "AI" Part)
Create a file named `train_real_yolo.py` and run it.

```python
from ultralytics import YOLO

# 1. Load the base model (pre-trained on COCO)
model = YOLO('yolov8n.pt') 

# 2. Train it on YOUR data
# 'data.yaml' is the file you got from Roboflow export
results = model.train(
    data='path/to/dataset/data.yaml', 
    epochs=50, 
    imgsz=640,
    project='jakarta_flood_model'
)

# 3. Save the result
# The new "smart" model will be in runs/detect/train/weights/best.pt
# Rename it to 'yolo_model.pt' and put it in the models/ folder.
```

## 4. Verify It
Run the verification script:
```bash
yolo detect predict model=models/yolo_model.pt source='path/to/test_image.jpg'
```
If it draws a box saying `flood_water 0.85`, congratulations! You have a legit AI. 🚀
