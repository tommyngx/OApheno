# YOLO26 OA Pose v4 HRNet

`yolo26-posev4.yaml` is an experimental pose model for dense medical landmark detection. It keeps the standard YOLO26 pose head and public keypoint output, but replaces the YOLO26 backbone with a clean-room HRNet-style high-resolution backbone adapter.

This variant is intended to answer one question: does an HRNet-style high-resolution backbone improve 129-point knee X-ray landmark accuracy over the standard YOLO26 pose backbone?

## Architecture

```text
Input
  -> HRNetYOLOBackbone
      -> P2 stride 4
      -> P3 stride 8
      -> P4 stride 16
      -> P5 stride 32
  -> YOLO26-style FPN/PAN neck
  -> original Pose26 head
  -> standard YOLO pose keypoints
```

For `896 x 896` input, `HRNetYOLOBackbone` returns:

- `P2`: `B x 128 x 224 x 224`
- `P3`: `B x 256 x 112 x 112`
- `P4`: `B x 512 x 56 x 56`
- `P5`: `B x 512 x 28 x 28`

v4 does not add SimCC, auxiliary heatmap supervision, GNN refinement, contour refinement, or a custom pose loss.

## Files

- Model YAML: `ultralytics/cfg/models/26oa/yolo26-posev4.yaml`
- Backbone: `ultralytics/nn/modules/hrnet_yolo_backbone.py`
- Parser registration: `ultralytics/nn/tasks.py`
- Smoke test: `tests/custom/test_yolo26_posev4_hrnet.py`

The HRNet adapter is a clean-room implementation based on the HRNet design concept. It does not copy code from the official HRNet repository and does not bundle pretrained weights. Passing `pretrained=True` raises `NotImplementedError`.

## Training

Full training:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev4.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
```

Safer first smoke training:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev4.yaml data=your_knee_pose.yaml imgsz=768 epochs=5 batch=2
```

HRNet plus P2 stride-4 features can be VRAM-heavy. Start with `imgsz=768` and a smaller batch size than the standard YOLO26 pose variants, then move to `imgsz=896` if memory allows.

## Evaluation

For the knee landmark task, compare v4 against the original YOLO26 pose model with medical metrics, not only COCO-style pose metrics:

- mean radial error in pixels
- normalized mean error using knee crop width or tibial width
- percentage of keypoints within 2, 4, and 8 pixels
- per-region error for femur, tibia, joint margin, and osteophyte-related points
- downstream B-score error
- downstream JSW measurement error
- per-image failure rate

## Export

The public output remains standard YOLO pose keypoints, but ONNX export should still be tested because the backbone returns multi-branch features before `Index` layers select P2/P3/P4/P5.
