# YOLO26 OA Pose v9 — Per-Region Query Refinement

V9 keeps the v1 backbone, neck, standard Pose26 prediction branches, RLE, and existing OA26 auxiliary losses. New
neural-network code is isolated in `ultralytics/nn/modules/oa26_region_refine/`; its schema, losses, and training plot
are isolated in `ultralytics/utils/oa26_region_refine/`.

## MESKO4GF2 annotation contract

The implementation was checked against `Reference/yolo_mesko4GF2/data.yaml`, `summary.json`, and every label row:

| Class | ID | Valid local points | Source point range | Padded pose shape |
| --- | ---: | ---: | --- | --- |
| femur | 0 | 45 | 0–44 | 51 × 3 |
| tibia | 1 | 51 | 45–95 | 51 × 3 |
| fibula | 2 | 24 | 96–119 | 51 × 3 |
| patella | 3 | 9 | 120–128 | 51 × 3 |

Consequently v9 uses `nc: 4` and `kpt_shape: [51, 3]`. Padding slots are masked from attention and every refinement
loss. The single source-of-truth mapping is `ultralytics/utils/oa26_region_refine/region_schema.py`.

## Refinement path

For each image, v9 selects the highest-confidence anchor independently for each of the four classes. Each selected
bone instance then follows its own path:

1. Its standard Pose26 box and coarse local keypoints are decoded.
2. ROIAlign extracts a padded `24 × 24` P4 feature map.
3. Coordinate, confidence, local point identity, and bone-class identity form landmark queries.
4. Cross-attention lets each query search the complete ROI.
5. Self-attention exchanges shape information only inside that instance row.
6. Query/image similarity produces a spatial probability map over the ROI.
7. Soft-argmax maps the result back to image pixels.

Femur, tibia, fibula, and patella rows never attend to one another. There is no patch-per-landmark extraction and no
bounded residual output.

## Public output compatibility

Raw branch dictionaries expose `coarse_region_kpts`, `refined_region_kpts`, `region_boxes`, `region_heatmaps`, class
IDs, masks, selected anchor indices, and ROI-normalized coordinates for debugging and loss calculation.

The public inference tensor remains the normal v1/Ultralytics pose layout:

```text
[x1, y1, x2, y2, confidence, class_id, 51 × (x, y, visibility)]
```

Its last dimension is therefore 159. V9 postprocessing chooses the refined pose corresponding to each returned
detection's class; downstream code does not need to handle v9-only channels.

## Losses

Four class-local losses are appended after the existing v1/RLE components:

- region heatmap loss
- region coordinate loss
- region neighbour loss
- region curvature loss

Ground truth is gathered using `(batch_idx, class_id)`. Neighbour and curvature masks operate independently per row,
so no structural edge crosses between bones or enters a padded slot.

## Per-epoch dashboard

For v9 training only, `runs/.../pose_detection_performance.png` is overwritten after every validation epoch. It is a
2 × 2 figure containing detection mAP50 and mAP50–95 on the top row, pose mAP50 and mAP50–95 on the bottom row. The
top three epochs are marked and annotated in every subplot.

## Train

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev9.yaml \
  data=/path/to/yolo_mesko4GF2/data.yaml imgsz=896 epochs=100
```

The reference copy's `path:` field points to its original machine location, so update that field before using the copy
under `Reference/` directly for training.

A v1 checkpoint trained with the same `nc=4`, `kpt_shape=[51,3]` dataset override loads with `strict=False`; only the
new `region_refine_head` weights are missing.
