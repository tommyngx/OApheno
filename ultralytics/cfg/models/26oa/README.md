# YOLO26 OA Pose Experiments

This folder contains experimental YOLO26 pose model variants for dense medical knee X-ray landmarks. The default target
is `nc: 1` and `kpt_shape: [129, 3]`.

All models keep standard YOLO pose prediction compatibility unless noted. Prediction, validation, export, and downstream
postprocessing should still receive normal YOLO pose keypoints.

## Model Summary

| Model | YAML | Backbone | Head | Extra loss | Output | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | `ultralytics/cfg/models/26oa/yolo26-posev1.yaml` | YOLO26-style P2-P5 | `OA26HeatmapPose` | Heatmap + coord + neighbour + curve | Standard pose | Test full auxiliary landmark supervision |
| v2 | `ultralytics/cfg/models/26oa/yolo26-posev2.yaml` | YOLO26-style P2-P5 | `OA26HeatmapPose` | Heatmap only | Standard pose | Isolate heatmap supervision effect |
| v3 | `ultralytics/cfg/models/26oa/yolo26-posev3.yaml` | YOLO26-style P2-P5 | `OA26SimCCPose` | SimCC x/y | Standard pose | Test coordinate-distribution supervision |
| v4 | `ultralytics/cfg/models/26oa/yolo26-posev4.yaml` | `HRNet` | `Pose26` | None | Standard pose | Isolate HRNet high-resolution backbone effect |
| v5 | `ultralytics/cfg/models/26oa/yolo26-posev5.yaml` | `ConvNeXtV2N` pretrained | `Pose26` | None | Standard pose | Isolate pretrained ConvNeXtV2-Nano backbone effect |

## Module Locations

All custom neural network modules for this experiment group live in `ultralytics/nn/modules/oa26/`:

- `pose_heads.py`: `OA26HeatmapPose`, `OA26SimCCPose`
- `hrnet.py`: `HRNet`
- `convnextv2_n.py`: `ConvNeXtV2N` for Nano
- `convnextv2_t.py`: `ConvNeXtV2T` for Tiny, kept for quick future swaps

## Quick Training Commands

Auxiliary-loss variants:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev1.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev2.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev3.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
```

Backbone-only variants:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev4.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev5.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
```

Safer smoke runs:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev1.yaml data=your_knee_pose.yaml imgsz=768 epochs=5 batch=2
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev5.yaml data=your_knee_pose.yaml imgsz=768 epochs=5 batch=2
```

Swap the model path in the smoke command to compare all five variants under the same data, image size, batch size, and
seed.

## Recommended Experiment Order

1. Build each model from YAML.
2. Run a small dummy forward smoke test.
3. Train for 5 epochs at `imgsz=768`, small batch.
4. Train the best candidates at `imgsz=896`.
5. Compare against original `ultralytics/cfg/models/26/yolo26-pose.yaml`.

## Memory Notes

- v1-v3 include a P2 stride-4 path and auxiliary branches, so they can use more VRAM than original YOLO26 pose.
- v4 HRNet and v5 ConvNeXtV2-Nano keep the standard `Pose26` head but still use P2 stride-4 features.
- Start with `imgsz=768` and small batch size before moving to `imgsz=896`.
- v5 requires `timm` for `convnextv2_nano` and downloads pretrained weights on first use if they are not cached.

## Medical Landmark Metrics

Do not rely only on COCO-style pose mAP for this task. Track:

- mean radial error in pixels
- normalized mean error using knee crop width or tibial width
- percentage of keypoints within 2, 4, and 8 pixels
- per-region landmark error: femur, tibia, joint margin, osteophyte-related points
- downstream B-score error
- downstream JSW measurement error
- per-image failure rate

## Export Notes

The public output is standard YOLO pose keypoints. Still test ONNX/export separately for v1-v5 because v1-v3 introduce
custom auxiliary branches and v4-v5 introduce custom backbone wrappers.
