# YOLO26 OA Pose v7/v8 Canonical HRNet

The v7 and v8 experiments replace the lightweight v4 adapter with the canonical HRNet-Pose layout while preserving the YOLO26 P2--P5 neck inputs and standard `Pose26` output.

| Variant | Backbone | Intended use |
| --- | --- | --- |
| v7 | HRNet-W32 | Balanced canonical HRNet baseline |
| v8 | HRNet-W48 | Accuracy-first comparison |

Both backbones use a stride-4 stem, four stage-1 bottleneck blocks, then multi-resolution stages with 1, 4 and 3 fusion modules. Each branch in a module uses four basic residual blocks. Branches are introduced progressively from one to four resolutions, and every module fuses all active resolutions.

The backbone is train-from-scratch only. It does not download or load pretrained weights. Output adapters retain the existing neck contract at P2/P3/P4/P5: `128/256/512/512` channels and strides `4/8/16/32`.

Train v7 first, then use v8 only if its medical landmark metrics justify the additional memory and inference cost:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev7.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev8.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
```
