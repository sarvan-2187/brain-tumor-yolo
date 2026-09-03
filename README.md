# Brain Tumor Detection: YOLOv8 vs YOLO11 vs YOLO26

Comparing three YOLO architectures on MRI brain tumor detection, using the
Kaggle [MRI for Brain Tumor with Bounding Boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)
dataset (4 classes: Glioma, Meningioma, Pituitary, No Tumor). All three
trained under a matched budget: `imgsz=640, epochs=50, seed=42`.

**Not a diagnostic tool.** Research comparison only.

## Results

### Nano (n) scale

| Model  | Scale     | mAP50  | mAP50-95 | Precision | Recall |
|--------|-----------|--------|----------|-----------|--------|
| YOLOv8 | yolov8n   | 0.9664 | 0.7808   | 0.9445    | 0.9483 |
| YOLO11 | yolo11n   | 0.9681 | 0.7752   | 0.9627    | 0.9333 |
| YOLO26 | yolo26n   | 0.9595 | 0.7678   | 0.9487    | 0.9346 |

Per-class AP50:

| Model  | Glioma | Meningioma | No Tumor | Pituitary |
|--------|--------|------------|----------|-----------|
| YOLOv8 | 0.9305 | 0.9939     | 0.9950   | 0.9460    |
| YOLO11 | 0.9313 | 0.9942     | 0.9950   | 0.9518    |
| YOLO26 | 0.9328 | 0.9927     | 0.9949   | 0.9175    |

### Small (s) scale

| Model  | Scale     | mAP50  | mAP50-95 | Precision | Recall |
|--------|-----------|--------|----------|-----------|--------|
| YOLOv8 | yolov8s   | 0.9621 | 0.7797   | 0.9643    | 0.9286 |
| YOLO11 | yolo11s   | 0.9664 | 0.7797   | 0.9618    | 0.9451 |
| YOLO26 | yolo26s   | 0.9639 | 0.7795   | 0.9667    | 0.9205 |

Per-class AP50:

| Model  | Glioma | Meningioma | No Tumor | Pituitary |
|--------|--------|------------|----------|-----------|
| YOLOv8 | 0.9373 | 0.9942     | 0.9850   | 0.9319    |
| YOLO11 | 0.9359 | 0.9944     | 0.9948   | 0.9407    |
| YOLO26 | 0.9247 | 0.9944     | 0.9949   | 0.9415    |

### Medium (m) scale

| Model  | Scale     | mAP50  | mAP50-95 | Precision | Recall |
|--------|-----------|--------|----------|-----------|--------|
| YOLOv8 | yolov8m   | 0.9650 | 0.7839   | 0.9456    | 0.9494 |
| YOLO11 | yolo11m   | 0.9698 | 0.7823   | 0.9398    | 0.9482 |
| YOLO26 | yolo26m   | 0.9676 | 0.7761   | 0.9602    | 0.9217 |

Per-class AP50:

| Model  | Glioma | Meningioma | No Tumor | Pituitary |
|--------|--------|------------|----------|-----------|
| YOLOv8 | 0.9493 | 0.9930     | 0.9950   | 0.9229    |
| YOLO11 | 0.9374 | 0.9935     | 0.9947   | 0.9535    |
| YOLO26 | 0.9412 | 0.9928     | 0.9949   | 0.9414    |

### Caveat

YOLO26 removes NMS and DFL and uses a different default training recipe
(MuSGD optimizer, ProgLoss/STAL) than YOLOv8/YOLO11. Matching
epochs/imgsz/seed alone does not make this a fully controlled architecture
comparison; read YOLO26's numbers with that in mind.

## Repo

- `train_yolov8.ipynb` / `train_yolo11.ipynb` / `train_yolo26.ipynb`: self-contained Colab notebooks, download the dataset, train, save `{model}_outputs/`.
- `comparison.ipynb`: builds the tables/charts above from the `{model}_outputs/` folders, no GPU needed.
- `app.py`: Streamlit demo, upload an MRI slice and run all trained models at once.
- `yolov8_outputs/` / `yolo11_outputs/` / `yolo26_outputs/`: weights, metrics JSON, training run per model.

## Setup

1. Train: open a `train_*.ipynb` in Colab, run all cells.
2. Compare: unzip each notebook's output next to `comparison.ipynb` as `{model}_outputs/`, run it.
3. Demo: `pip install -r requirements-yolo.txt && streamlit run app.py` from the folder containing the `{model}_outputs/` directories.
