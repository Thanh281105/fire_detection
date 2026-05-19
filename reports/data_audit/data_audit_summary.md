# YOLO11 Data Audit

Dataset: `datasets\fire_vn_yolo11seg_v1`

## Overview

- Images: `15615`
- Objects: `31134`
- Background images: `4229`
- Background ratio: `27.08%`

## Class Counts

- `smoke`: `13645`
- `fire`: `17489`

## Split And Group Counts

- `train g01`: 7873 images, 20118 objects, 0 background
- `train g02`: 1803 images, 4228 objects, 208 background
- `train g03`: 562 images, 0 objects, 562 background
- `train g04`: 606 images, 2197 objects, 0 background
- `train g05`: 2981 images, 23 objects, 2975 background
- `valid g01`: 633 images, 1964 objects, 0 background
- `valid g02`: 44 images, 171 objects, 0 background
- `valid g03`: 20 images, 0 objects, 20 background
- `valid g04`: 20 images, 79 objects, 0 background
- `valid g05`: 240 images, 0 objects, 240 background
- `test g01`: 543 images, 2056 objects, 0 background
- `test g02`: 44 images, 210 objects, 0 background
- `test g03`: 20 images, 0 objects, 20 background
- `test g04`: 20 images, 86 objects, 0 background
- `test g05`: 206 images, 2 objects, 204 background

## Near-Duplicate Audit

- Clusters: `1095`
- Images that would be removed: `1649`
- Removed by group: `{'01': 1518, '02': 20, '03': 9, '05': 102}`

Review contact sheets in `near_duplicates/` before enabling near-duplicate removal.

## Generated Artifacts

- `figures/`: charts for report writing.
- `samples/`: annotated image montages.
- `near_duplicates/`: visual duplicate audit sheets.
- `tables/`: CSV tables for appendix or further analysis.
