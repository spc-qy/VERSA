# VERSA

A View-Adaptive Spatial Refinement Framework for Sheep Thoracolumbar Vertebral Formula Classification

VERSA is a deep learning framework for classifying sheep thoracolumbar vertebral formula from radiographic images. The framework integrates multi-view radiographic inputs, heatmap-based anatomical priors, view-adaptive attention fusion, and structural feature refinement to improve the recognition of thoracic and lumbar vertebral formula patterns. It is designed for radiographic phenotyping tasks in livestock, especially for distinguishing thoracolumbar formula classes from two-view or multi-view X-ray images.

![VERSA workflow](docs/figures/versa_workflow.jpg)

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

## Installation

Create a conda environment:

```bash
conda create -n versa python=3.10 -y
conda activate versa
```

Install required packages:

```bash
pip install -r requirements.txt
```

## Data organization

The input data directory should be organized as follows:

```text
DATA_ROOT/
├── AB/
│   ├── sample_001/
│   │   ├── 1.png
│   │   └── 2.png
│   ├── sample_002/
│   │   ├── 1.png
│   │   └── 2.png
│   └── label/
│       └── sample_003/
│           ├── 1.png
│           └── 2.png
├── BA/
│   ├── sample_001/
│   │   ├── 1.png
│   │   └── 2.png
│   ├── sample_002/
│   │   ├── 1.png
│   │   └── 2.png
│   └── label/
│       └── sample_003/
│           ├── 1.png
│           └── 2.png
└── _priors/
    ├── AB/
    │   └── sample_001/
    │       ├── 1_heat.png
    │       └── 2_heat.png
    └── BA/
        └── sample_001/
            ├── 1_heat.png
            └── 2_heat.png
```

Each sample folder should contain two radiographic views named `1.png` and `2.png`.

The heatmap prior files are optional when `FUSION_ALPHA=1.0`. They are used when the structure-enhanced fusion branch is enabled with `FUSION_ALPHA < 1.0`.


## Run

Edit paths and parameters in:

```text
configs/config.py
```

Then run:

```bash
python train.py
```
