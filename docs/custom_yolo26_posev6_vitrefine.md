# YOLO26 OA Pose v6 ViTRefine

`yolo26-posev6.yaml` is an experimental pose model for dense medical knee X-ray landmarks. It keeps the YOLO26/OA P2-P5 pose pipeline and standard `Pose26` output, but inserts a small ViTPose-inspired transformer refinement block on P4/16.

The experiment asks one narrow question: does global token reasoning on a mid-level feature map improve 129-point knee landmark localization?

## Architecture

```text
Input
-> YOLO26-style backbone
-> YOLO26-style P2/P3/P4/P5 neck
-> ViTRefine on P4/16
-> Pose26(P2, P3, P4, P5)
-> standard YOLO pose keypoints
```

`ViTRefine` borrows only the global token mixing idea from ViTPose. It does not implement full ViTPose, ViTPose++ MoE, SimCC, heatmap auxiliary supervision, HRNet, ConvNeXtV2, a new trainer, or a new pose loss.

## Placement

The refinement block is placed on P4/16 after the top-down P4 fusion layer. At `imgsz=896`, P4 has `56 x 56 = 3136` tokens, which is practical for a small transformer block. P2/4 is left as convolutional features for fine landmark localization because full attention at P2 would be too expensive.

## Commands

Full training:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev6.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
```

Safer smoke run:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev6.yaml data=your_knee_pose.yaml imgsz=768 epochs=5 batch=2
```

## Practical Notes

- Start with `imgsz=768` and small batch size.
- Use the default `ViTRefine` depth `1` first.
- If VRAM is high, lower `imgsz`, reduce `embed_dim`, reduce depth, or keep refinement only on P4.
- Compare with v2, v4, v5, and original `ultralytics/cfg/models/26/yolo26-pose.yaml` under the same seed and training schedule.

## Medical Metrics

Track medical landmark metrics in addition to pose mAP:

- mean radial error in pixels
- normalized mean error by knee crop width or tibial width
- percentage of keypoints within 2, 4, and 8 pixels
- per-region error: femur, tibia, joint margin, osteophyte-related points
- downstream B-score error
- downstream JSW error
- per-image failure rate
