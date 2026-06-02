# sheep-vertebra

Code for sheep thoracolumbar vertebral formula classification using radiographic images.

## Project structure

```text
sheep-vertebra/
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
