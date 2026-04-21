# Algorithms

- [HRNet, ViTPose, & RTMPose (2D)](#hrnet-vitpose--rtmpose-2d-pose-estimation)
- [MotionBERT (3D lifting)](#motionbert-3d-body-pose-lifting)
- [Multiview_eth_xgaze](#multiview_eth_xgaze)
- [Py-FEAT](#pyfeat)
- [SPIGA](#spiga)

<br>


## HRNet, ViTPose, & RTMPose (2D Pose Estimation)

The NICE toolbox integrates the **MMPose framework** to provide a suite of robust **top-down 2D human pose estimation** models. These algorithms operate by first using an object detector (e.g., Faster R-CNN) to extract bounding boxes around subjects, followed by dedicated keypoint localization on those cropped regions. The toolbox supports a range of architectures, from high-precision transformers to real-time optimized CNNs.

### High-Resolution Networks
- **HRNet (w48)**: A convolutional neural network designed to maintain high-resolution representations throughout the entire inference process. Instead of encoding images into low-resolution feature maps and decoding them later, HRNet connects high-to-low resolution subnetworks in parallel with repeated multi-scale fusion. This preserves fine spatial details and excels at localizing complex or occluded joints.   
  [Sun et al., 2019](https://arxiv.org/abs/1902.09212)

### Vision Transformers
- **ViTPose / ViTPose-Huge**: A state-of-the-art baseline that employs plain, non-hierarchical Vision Transformers (ViTs) as the feature extraction backbone. By leveraging global self-attention mechanisms, ViTPose efficiently models the long-range relationships between different anatomical joints, offering superior robustness against severe occlusions and challenging body variations.   
  [Xu et al., 2022](https://arxiv.org/abs/2204.12484)

### Real-Time Pose Estimation
- **RTMPose (m/l/wholebody)**: A framework explicitly optimized for real-time multi-person pose estimation. RTMPose utilizes a highly efficient CSPNeXt backbone and replaces traditional heatmap generation with a SimCC-based prediction head (treating keypoint localization as a classification task). This architectural shift delivers an exceptional balance between low latency and high accuracy.   
  [Jiang et al., 2023](https://arxiv.org/abs/2303.07399)


## MotionBERT (3D body pose lifting)

**MotionBERT** is a **2D-to-3D pose lifter** based on a Dual-stream Spatio-temporal Transformer (DSTformer), integrated through **MMPose**. It lifts per-frame 2D keypoints into **root-relative 3D poses** without requiring a second camera calibration or world-space alignment pass.

In NICE Toolbox, MotionBERT runs as the **`motionbert`** algorithm under the **`body_joints_lifted`** component. It reads 2D keypoints and bounding boxes directly from an existing **`body_joints`** NPZ (default upstream: **`vitpose_huge`**), so subject ordering and tracking stay consistent with the 2D pipeline. The lifter outputs **17 keypoints in Human3.6M order**.

Post-processing applies optional temporal filtering and confidence masking on the 2D and 3D arrays. The upstream algorithm can be configured via **`input_detector_names`** in [`detectors_config.toml`](../../configs/detectors_config.toml).

[Zhu et al., MotionBERT, ICCV 2023](https://arxiv.org/abs/2210.06551) · [MMPose MotionBERT model zoo](https://github.com/open-mmlab/mmpose/tree/main/configs/body_3d_keypoint/motionbert)


## Multiview_eth_xgaze

COMING SOON...


## Py-FEAT (Facial Expression Analysis Toolbox)

Py-FEAT includes a variety of **pre-trained models** for **face detection, facial landmark tracking, action unit (AU) recognition, emotion detection, and identity verification**. These models enable automated facial expression analysis. In the NICE toolbox, we only use *face detection, action unit (AU) recognition, emotion detection, and identity verification**. The associated algorithms are listed below.

### Face Detection & Pose Estimation
- **img2pose**: A one-shot model for simultaneous **face detection** and **6DoF head pose estimation**.  
  [Albiero et al., 2020](https://arxiv.org/pdf/2012.07791v2)

### Action Unit (AU) Detection
- **xgb** (**default**): An **XGBoost classifier** trained on multiple facial expression datasets (BP4D, DISFA, CK+, etc.). It provides **continuous AU probabilities**, except for AU07, which is optimized for **binary detection**.

### Emotion Detection
- **resmasknet** (**default**): A **deep learning model** trained for facial expression recognition using a **Residual Masking Network**.  
  [Pham et. al., 2020](https://ieeexplore.ieee.org/document/9411919)

### Identity Detection
- **facenet**: A **face recognition model** based on **Inception-ResNet (V1)**, pretrained on **VGGFace2 and CASIA-Webface**.  
  [Schroff et al., 2015](https://arxiv.org/abs/1503.03832)

## SPIGA (Shape Preserving Facial Landmarks with Graph Attention)

**SPIGA** is a **state-of-the-art face alignment and head pose estimation model** that combines **CNNs and Graph Neural Networks (GNNs)** to predict stable facial landmarks under challenging conditions (e.g., occlusions, expressions, pose). In the NICE toolbox, SPIGA is used for **landmark localization** and **6DoF head pose estimation**. It operates on multi-camera image sequences and produces vectorized nose origin and orientation data for each visible subject.

### Landmark Localization & Head Pose Estimation
- **SPIGA** (**default**): A hybrid **CNN-GNN** model trained for **dense face alignment** and **pose estimation**, achieving top performance on WFLW, COFW, and 300W benchmarks.  
  [Prados-Torreblanca et al., 2022](https://arxiv.org/abs/2210.07233)

SPIGA uses **InsightFace** for face detection, then applies its GNN-powered inference module to extract facial landmarks and head orientation vectors. Outputs include annotated images (if enabled) and compressed `.npz` files containing head pose vectors for each camera-subject-frame triplet.



