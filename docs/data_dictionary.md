# VERSA Dataset Dictionary

## Dataset version

Dataset version: **1.0**

This document describes the radiographic dataset used for the development and
evaluation of VERSA.

## Data availability

The radiographic data are available from:

http://bioinfor.imu.edu.cn/biocloud/downloads/Thoracolumbar%20Vertebral.zip

The raw radiographic images are not hosted in this GitHub repository.

## Dataset overview

The complete study dataset contains **4,542 radiographic images from 1,205 sheep**.

Five radiographic classification tasks were evaluated. Task-specific sample
counts are not mutually exclusive because an individual animal could contribute
to more than one classification task.

| Task | Classes | Sample size |
|---|---|---:|
| Single-image thoracic vertebral number classification | T13, T14 | 691 images |
| Single-image lumbar vertebral number classification | L6, L7 | 1,017 images |
| Single-image thoracolumbar formula classification | T13L6, T13L7, T14L6 | 270 images |
| Two-view thoracolumbar formula classification | T13L6, T13L7, T14L6, T14L7 | 686 animals / 1,372 images |
| Three-view thoracolumbar formula classification | T13L6, T13L7, T14L6, T14L7 | 518 animals / 1,554 images |

## Class definitions

| Formula label | Abbreviation | Thoracic vertebrae | Lumbar vertebrae |
|---|---|---:|---:|
| T13L6 | AA | 13 | 6 |
| T13L7 | AB | 13 | 7 |
| T14L6 | BA | 14 | 6 |
| T14L7 | BB | 14 | 7 |

For isolated vertebral-number classification:

- **T13**: 13 thoracic vertebrae
- **T14**: 14 thoracic vertebrae
- **L6**: 6 lumbar vertebrae
- **L7**: 7 lumbar vertebrae

## Task-specific class counts

### Single-image thoracic vertebral number classification

| Class | Number of images |
|---|---:|
| T13 | 509 |
| T14 | 182 |

### Single-image lumbar vertebral number classification

| Class | Number of images |
|---|---:|
| L6 | 432 |
| L7 | 585 |

### Single-image thoracolumbar formula classification

| Class | Number of images |
|---|---:|
| T13L6 (AA) | 59 |
| T13L7 (AB) | 156 |
| T14L6 (BA) | 55 |

### Two-view thoracolumbar formula classification

| Class | Number of animals |
|---|---:|
| T13L6 (AA) | 125 |
| T13L7 (AB) | 376 |
| T14L6 (BA) | 167 |
| T14L7 (BB) | 18 |

Each animal contributes two radiographs to this task.

### Three-view thoracolumbar formula classification

| Class | Number of animals |
|---|---:|
| T13L6 (AA) | 97 |
| T13L7 (AB) | 273 |
| T14L6 (BA) | 122 |
| T14L7 (BB) | 26 |

Each animal contributes three radiographs to this task.

## Radiographic view definitions

The dataset contains radiographs covering different parts of the
thoracolumbar region.

- **Thoracic view**: radiograph primarily covering the thoracic vertebral region.
- **Thoracolumbar-junction view**: radiograph covering the transition between
  the thoracic and lumbar regions.
- **Lumbar view**: radiograph primarily covering the lumbar vertebral region.

In the two-view task, thoracic and lumbar radiographs were used.

In the three-view task, thoracic, thoracolumbar-junction, and lumbar
radiographs were used when two radiographs were insufficient for complete
thoracolumbar coverage.

## Anatomical landmark annotations and heatmap priors

For a subset of training radiographs, vertebral landmarks were manually
annotated and converted into Gaussian-smoothed anatomical heatmaps.

The anatomical heatmaps were used as spatial priors and as the basis for
heatmap-derived structural descriptors.

Manual heatmap priors were restricted to training data. Inner-validation and
outer-test samples were evaluated without manually generated heatmap priors.

## Cross-validation metadata

Exact animal-level outer-fold assignments for the five classification tasks
are provided separately in the `splits/` directory of this repository.

All radiographs from the same sheep were assigned to the same data partition
within a given classification task.

## License

The MIT License in this GitHub repository applies to the VERSA source code.

The radiographic dataset is distributed separately through the data repository
listed above and is not covered by the software license in this GitHub
repository.
