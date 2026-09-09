# VERSA

**A View-Adaptive Spatial Refinement Framework for Sheep Thoracolumbar Vertebral Formula Classification**

**Code version: 1.0.0**

VERSA is a deep learning framework for classifying sheep thoracolumbar vertebral formula from radiographic images. The framework integrates multi-view radiographic inputs, heatmap-based anatomical priors, view-adaptive attention fusion, and structural feature refinement to improve the recognition of thoracic and lumbar vertebral formula patterns. It is designed for radiographic phenotyping tasks in livestock, especially for distinguishing thoracolumbar formula classes from two-view or multi-view X-ray images.

![VERSA workflow](docs/figures/versa_workflow.jpg)

## Project structure

```text
VERSA/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── config.py
├── docs/
│   ├── data_dictionary.md
│   └── figures/
├── src/
│   ├── dataset.py
│   ├── heatmap_features.py
│   ├── model.py
│   ├── train_utils.py
│   ├── metrics.py
│   └── plot_utils.py
└── train.py
```

## Environment

The reported experiments were run using Python 3.12.12.
Package versions required for reproducibility are provided in
[`requirements.txt`](requirements.txt).

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

## Data availability

The radiographic dataset used in this study is available from:

http://bioinfor.imu.edu.cn/biocloud/downloads/Thoracolumbar%20Vertebral.zip

**Dataset version: 1.0**

The complete study dataset contains 4,542 radiographic images from 1,205 sheep.

The raw radiographic images are not hosted in this GitHub repository.
Definitions of the vertebral formulae, classification tasks, radiographic
views, and task-specific sample counts are provided in
[`docs/data_dictionary.md`](docs/data_dictionary.md).

The MIT License of this repository applies to the VERSA source code and does
not apply to the radiographic dataset.

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


## Usage

Run the image-only setting:

```bash
DATA_ROOT="/path/to/your/data" \
FUSION_ALPHA=1.0 \
SAVE_DIR="/path/to/save/results" \
python train.py
```

Run the structure-enhanced fusion setting:

```bash
DATA_ROOT="/path/to/your/data" \
FUSION_ALPHA=0.8 \
SAVE_DIR="/path/to/save/results" \
python train.py
```

`FUSION_ALPHA` controls the fusion weight between image features and structural features:

```text
FUSION_ALPHA = 1.0  -> image-only path
FUSION_ALPHA < 1.0  -> image + structural feature fusion path
```

## Output

The results will be saved to `SAVE_DIR`, including:

```text
*_results.csv
*_roc.csv
*_prc.csv
```

These files contain performance results from animal-level stratified five-fold
cross-validation with an internal validation split, together with ROC and
precision-recall curve data.

## Citation

Citation information will be added after manuscript publication.

## License

The VERSA source code is released under the MIT License. See
[`LICENSE`](LICENSE) for details.

The software license does not apply to the radiographic dataset, which is
distributed separately through the data repository described above.
