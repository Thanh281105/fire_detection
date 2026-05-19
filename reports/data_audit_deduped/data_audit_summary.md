# YOLO11 Data Audit

Dataset: `datasets\fire_vn_yolo11seg_v1`

## Overview

- Images: `13896`
- Objects: `28279`
- Background images: `4099`
- Background ratio: `29.50%`

## Class Counts

- `smoke`: `12356`
- `fire`: `15923`

## Split And Group Counts

- `train g01`: 6552 images, 18479 objects, 0 background
- `train g02`: 1711 images, 4498 objects, 188 background
- `train g03`: 555 images, 0 objects, 555 background
- `train g04`: 612 images, 2086 objects, 0 background
- `train g05`: 2892 images, 23 objects, 2885 background
- `valid g01`: 527 images, 1511 objects, 0 background
- `valid g02`: 42 images, 105 objects, 0 background
- `valid g03`: 19 images, 0 objects, 19 background
- `valid g04`: 20 images, 108 objects, 0 background
- `valid g05`: 233 images, 0 objects, 233 background
- `test g01`: 452 images, 1245 objects, 0 background
- `test g02`: 42 images, 132 objects, 0 background
- `test g03`: 19 images, 0 objects, 19 background
- `test g04`: 20 images, 92 objects, 0 background
- `test g05`: 200 images, 0 objects, 200 background

## Duplicate Preprocessing Applied

- Duplicate clusters: `1095`
- Removed source images: `1649`
- Kept source images in duplicate clusters: `1095`
- Removed by group: `{'01': 1518, '02': 20, '03': 9, '05': 102}`

Audit file: `../datasets/fire_vn_yolo11seg_v1/metadata/duplicate_sources.csv`

## Generated Artifacts

- `figures/`: charts for report writing.
- `samples/`: annotated image montages.
- `near_duplicates/`: visual duplicate audit sheets.
- `tables/`: CSV tables for appendix or further analysis.
