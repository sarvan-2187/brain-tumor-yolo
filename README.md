# Brain Tumor Detection: YOLOv8 vs YOLO11 vs YOLO26

Comparing YOLOv8, YOLO11, and YOLO26 on MRI brain tumor detection, using the
Kaggle [MRI for Brain Tumor with Bounding Boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)
dataset (4 classes: Glioma, Meningioma, Pituitary, No Tumor).

The historical tables are retained under the historical-results section.
For the latest nano/small/medium comparison across all nine 100-epoch runs, use
[`yolo_notebooks/comparison.ipynb`](./yolo_notebooks/comparison.ipynb). That
notebook pins the intended result files, checks common evaluation settings,
and produces the current tables and figures shown below without mixing the
two experiments.

**Not a diagnostic tool.** Research comparison only.

## Current comparison: nine 100-epoch runs

The current comparison uses the YOLOv8, YOLO11, and YOLO26 nano (`n`), small
(`s`), and medium (`m`) models. The notebook checks that the runs record the
same dataset YAML, validation split, class order, image size (640), epoch
budget (100), batch size (16), and seed (42). It displays the recorded
optimizer setting and worker count as well; those are not proof that the
training procedures were otherwise identical.

### Complete performance results

Precision, recall, and AP values are scores from 0 to 1. The table is rounded
to four decimal places; the linked CSV retains the full recorded precision.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | AP50 Glioma | AP50 Meningioma | AP50 No Tumor | AP50 Pituitary | Inference (ms/image) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 0.9469 | 0.9501 | 0.9640 | 0.7789 | 0.9325 | 0.9924 | 0.9950 | 0.9362 | 4.006 |
| YOLO11n | 0.9630 | 0.9437 | 0.9682 | 0.7822 | 0.9333 | 0.9939 | 0.9950 | 0.9507 | 3.209 |
| YOLO26n | 0.9601 | 0.9412 | 0.9670 | 0.7757 | 0.9320 | 0.9932 | 0.9947 | 0.9480 | 4.329 |
| YOLOv8s | 0.9468 | 0.9437 | 0.9664 | 0.7838 | 0.9457 | 0.9934 | 0.9949 | 0.9317 | 5.884 |
| YOLO11s | 0.9723 | 0.9245 | 0.9668 | 0.7832 | 0.9433 | 0.9925 | 0.9949 | 0.9365 | 5.981 |
| YOLO26s | 0.9576 | 0.9507 | 0.9700 | 0.7848 | 0.9373 | 0.9942 | 0.9950 | 0.9533 | 6.332 |
| YOLOv8m | 0.9609 | 0.9262 | 0.9636 | 0.7797 | 0.9258 | 0.9943 | 0.9848 | 0.9495 | 13.522 |
| YOLO11m | 0.9681 | 0.9344 | 0.9687 | 0.7862 | 0.9385 | 0.9923 | 0.9945 | 0.9494 | 13.569 |
| YOLO26m | 0.9550 | 0.9444 | 0.9650 | 0.7794 | 0.9270 | 0.9940 | 0.9950 | 0.9439 | 13.398 |

### Training and model details

All nine runs completed 100 epochs using `imgsz=640`, `batch=16`, and
`seed=42`. They recorded the same [`_yolo_data/data.yaml`](./_yolo_data/data.yaml)
dataset config and `val` evaluation split. The best epoch below is the saved
run's recorded best-validation epoch. `optimizer=auto` is the recorded
setting, not a guarantee of the same optimizer implementation across model
families.

| Model | Epochs completed | Best epoch | Early stopped | Image size | Batch | Patience | Seed | Optimizer setting | Workers | AMP | Eval split | Params (M) | GFLOPs |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|---|---|---:|---:|
| YOLOv8n | 100 | 70 | No | 640 | 16 | 101 | 42 | auto | 0 | Yes | val | N/A | N/A |
| YOLO11n | 100 | 98 | No | 640 | 16 | 101 | 42 | auto | 2 | Yes | val | N/A | N/A |
| YOLO26n | 100 | 62 | No | 640 | 16 | 100 | 42 | auto | 0 | Yes | val | N/A | N/A |
| YOLOv8s | 100 | 76 | No | 640 | 16 | 101 | 42 | auto | 0 | Yes | val | N/A | N/A |
| YOLO11s | 100 | 70 | No | 640 | 16 | 101 | 42 | auto | 2 | Yes | val | N/A | N/A |
| YOLO26s | 100 | 70 | No | 640 | 16 | 100 | 42 | auto | 0 | Yes | val | N/A | N/A |
| YOLOv8m | 100 | 80 | No | 640 | 16 | 101 | 42 | auto | 0 | Yes | val | N/A | N/A |
| YOLO11m | 100 | 90 | No | 640 | 16 | 101 | 42 | auto | 2 | Yes | val | N/A | N/A |
| YOLO26m | 100 | 84 | No | 640 | 16 | 100 | 42 | auto | 0 | Yes | val | N/A | N/A |

Parameter counts and GFLOPs were not recorded in the source result files, so
they are marked N/A rather than estimated.

### Recorded validation timing breakdown

The following timing values are in milliseconds per image as recorded in the
result JSON files. The `loss` column is the validation loss-computation timing
reported by the framework; it is not a detection quality metric.

| Model | Preprocess (ms/image) | Inference (ms/image) | Loss (ms/image) | Postprocess (ms/image) |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.363 | 4.006 | 0.002 | 1.269 |
| YOLO11n | 0.799 | 3.209 | 0.001 | 0.747 |
| YOLO26n | 0.325 | 4.329 | 0.002 | 1.267 |
| YOLOv8s | 0.302 | 5.884 | 0.001 | 1.164 |
| YOLO11s | 0.786 | 5.981 | 0.001 | 0.753 |
| YOLO26s | 0.306 | 6.332 | 0.002 | 1.211 |
| YOLOv8m | 0.291 | 13.522 | 0.001 | 1.181 |
| YOLO11m | 0.671 | 13.569 | 0.001 | 0.632 |
| YOLO26m | 0.304 | 13.398 | 0.001 | 1.221 |

These are recorded timings, not a controlled deployment benchmark; compare
them cautiously because hardware and runtime conditions affect latency.

### Generated comparison files

Running all cells in [`yolo_notebooks/comparison.ipynb`](./yolo_notebooks/comparison.ipynb)
creates these files under [`paper_figures/`](./paper_figures/):

- [Full metrics table (CSV)](./paper_figures/model_comparison_100ep.csv)
- [mAP by scale](./paper_figures/map_comparison_100ep.png)
- [Per-class AP50 heatmap](./paper_figures/per_class_ap50_100ep.png)
- [mAP versus recorded inference latency](./paper_figures/accuracy_latency_100ep.png)
- [Validation mAP training curves](./paper_figures/validation_curves_100ep.png)

The notebook needs the nine result JSONs, their run `args.yaml` files, and
training `results.csv` logs listed in its model registry. It runs on CPU and
does not retrain the models.

### Why an older training-curve plot may be blank

The previous comparison code searched for every curve at
`{model}_outputs/runs/results.csv` and silently skipped a model when that path
did not exist. The actual runs are stored in named folders such as
`runs_100ep_workers0_20261113_batch16`, and some output folders also include
the run date. With no CSV loaded, Matplotlib showed empty axes with default
0-to-1 scales and blank legends; this was a path mismatch, not a model result
of zero. The current notebook maps each run explicitly and raises a
`FileNotFoundError` if an expected curve log is missing.

### Interpretation limits

- Each model has one run and one seed; there are no confidence intervals or
  significance tests.
- These are validation-set scores, not an independent test-set estimate.
- Matching the listed settings does not isolate architecture effects. YOLO
  families can differ in optimizer behavior, losses, augmentation, and other
  training defaults; YOLO26 in particular uses a different training recipe.
- Inference latency depends on hardware and runtime conditions and is not a
  standardized benchmark.
- These dataset metrics are for research comparison only, not clinical use.

## Historical results: 50-epoch snapshot

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
comparison; read YOLO26's numbers with that in mind. All reported values are
single-run validation results, not an independent test-set estimate.

## Repository guide

- `train_yolov8.ipynb` / `train_yolo11.ipynb` / `train_yolo26.ipynb`: self-contained Colab notebooks, download the dataset, train, save `{model}_outputs/`.
- [`yolo_notebooks/comparison.ipynb`](./yolo_notebooks/comparison.ipynb): compares the nine pinned 100-epoch results, checks the evaluation protocol, and exports a CSV and figures to `paper_figures/`; no GPU needed.
- `app.py`: Streamlit demo, upload an MRI slice and run all trained models at once.
- `yolov8_outputs/` / `yolo11_outputs/` / `yolo26_outputs/`: weights, metrics JSON, training run per model.

## Setup

1. Train: open a `train_*.ipynb` in Colab, run all cells.
2. Compare: ensure the nine result JSONs, their run `args.yaml` files, and `results.csv` training logs listed in [`comparison.ipynb`](./yolo_notebooks/comparison.ipynb) are present under the project folder, then run all notebook cells. The CSV and PNG results will be written to `paper_figures/`.
3. Demo: `pip install -r requirements-yolo.txt && streamlit run app.py` from the folder containing the `{model}_outputs/` directories.
