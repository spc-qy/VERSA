# VERSA

A View-Adaptive Spatial Refinement Framework for Sheep Thoracolumbar Vertebral Formula Classification. VERSA is a deep learning framework for classifying sheep thoracolumbar vertebral formula from radiographic images. The framework integrates multi-view radiographic inputs, heatmap-based anatomical priors, view-adaptive attention fusion, and structural feature refinement to improve the recognition of thoracic and lumbar vertebral formula patterns. It is designed for radiographic phenotyping tasks in livestock, especially for distinguishing thoracolumbar formula classes from two-view or multi-view X-ray images.

## Project structure

```text
VERSA/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.py
├── src/
│   ├── dataset.py
│   ├── heatmap_features.py
│   ├── model.py
│   ├── train_utils.py
│   ├── metrics.py
│   └── plot_utils.py
└── train.py
```

## Run

Edit paths and parameters in:

```text
configs/config.py
```

Then run:

```bash
python train.py
```
