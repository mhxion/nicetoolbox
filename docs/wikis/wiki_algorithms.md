# Algorithms

- [HRNet, ViTPose, & RTMPose (2D)](#hrnet-vitpose--rtmpose-2d-pose-estimation)
- [MotionBERT (3D lifting)](#motionbert-3d-body-pose-lifting)
- [SAM 3D Body](#sam-3d-body)
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

In NICE Toolbox, MotionBERT runs as the **`motionbert`** algorithm under the **`body_joints_local`** component. It reads 2D keypoints and bounding boxes directly from an existing **`body_joints`** NPZ (default upstream: **`vitpose_huge`**), so subject ordering and tracking stay consistent with the 2D pipeline. The lifter outputs **17 keypoints in Human3.6M order**.

Post-processing applies optional temporal filtering and confidence masking on the 2D and 3D arrays. The upstream algorithm can be configured via **`input_detector_names`** in [`detectors_config.toml`](../../configs/detectors_config.toml).

[Zhu et al., MotionBERT, ICCV 2023](https://arxiv.org/abs/2210.06551) · [MMPose MotionBERT model zoo](https://github.com/open-mmlab/mmpose/tree/main/configs/body_3d_keypoint/motionbert)

## SAM 3D Body

**SAM 3D Body** ([`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3)) estimates **3D human shape and pose** in **MHR** (Mesh Human Recovery) parameterization: 2D/3D keypoints, optional mesh vertices, and body/hand/shape parameters. In the NICE Toolbox it is the **`sam_3d_body`** **algorithm** (code under `method_detectors/sam_3d_body`); it writes **`body_joints/sam_3d_body/sam_3d_body.npz`** only when **intrinsics + projection matrix** exist for every configured camera (world-primary **`3d`** via **calibration alignment**, **not** classical multi-view ray triangulation). It **always** writes **`body_joints_local/sam_3d_body/sam_3d_body_camera.npz`** (camera-native **`3d`** primary) and **`body_mesh/sam_3d_body/sam_3d_body.npz`**. Schedule it by listing **`sam_3d_body`** under **`body_joints`**, **`body_joints_local`**, and **`body_mesh`** in **`[component_algorithm_mapping]`** and including those components in the run’s **`components`** list (see [`detectors_run_file.toml`](../../configs/detectors_run_file.toml)).

- **Environment**: Inference runs in a **separate** venv at **`./envs/sam_3d_body`** (`env_name = "venv:sam_3d_body"`). From the repo root run **`make install_sam3d_body`** to create the venv and **`pip install -r …/sam_3d_body_pip_requirements.txt`**. Upstream inference code comes from the git submodule **`submodules/sam-3d-body`** ([`OSLabTools/sam-3d-body`](https://github.com/OSLabTools/sam-3d-body)); see that repo’s **INSTALL.md** for stack details. The main `nicetoolbox` process only runs post-processing and visualization.
- **Cameras**: The repo template sets **`[algorithms.sam_3d_body].camera_names`** the same placeholders as MMPose (`<cur_cam_top>`, `<cur_cam_front>`), resolved from **`cam_top`** / **`cam_front`** in [`dataset_properties.toml`](../../configs/dataset_properties.toml) for the current dataset. For **communication_multiview** that is **view_top** and **view_center** only—not **view_left** / **view_right**. Override **`camera_names`** if you need a different pair.
- **Detectron2**: Required in the **`sam_3d_body`** venv (SAM 3D imports it). It is **not** listed as a plain pip dependency because wheels must match **PyTorch + CUDA**; see the bottom of **`sam_3d_body_pip_requirements.txt`** for example **`pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/...`** lines, or [Detectron2 install](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). **`nvcc`** must match **`torch.version.cuda`** (**12.1** / **`cu121`** in this repo’s pin). Align **`CUDA_HOME`** / **`PATH`** and the **host compiler** with NVIDIA’s guides for that toolkit (see [installation.md § SAM 3D Body](../installation.md#sam-3d-body-and-detectron2-cuda-toolkit-121)).
- **NumPy**: Keep the inference venv on **NumPy 1.x** (`numpy>=1.22,<2` in `sam_3d_body_pip_requirements.txt`) so raw **`.npz`** **`dtype=object`** payloads unpickle cleanly alongside a NumPy 1.x main `nicetoolbox` env.
- **Weights**: Loaded from **Hugging Face**; set `hugging_face_token` in **`machine_specific_paths.toml`**. Weights are not distributed via Keeper in the default asset manifest.
- **Outputs**: The subprocess writes a **staging** raw pack **`…/body_joints/sam_3d_body/_sam_3d_body_inference_raw.npz`**. Optional **JSON/CSV** dumps land under **`body_joints/sam_3d_body/detector_output/`**. Post-process writes **`sam_3d_body_camera.npz`** under **`body_joints_local/sam_3d_body/`**; **`sam_3d_body.npz`** under **`body_joints/sam_3d_body/`** only with usable calibration (intrinsics + projection matrix per camera); **`sam_3d_body.npz`** under **`body_mesh/sam_3d_body/`** (`faces`, **`vertices`**, optional **`vertices_world`**). See **`data_description.sam_3d_body.export_policy`**. Hugging Face cache defaults to **`nicetoolbox/detectors/assets/sam_3d_body/hf_home`**. If **`sam3d_repo_path`** is empty, inference uses **`submodules/sam-3d-body`** (initialize with **`git submodule update --init submodules/sam-3d-body`**).
- **Post-processing** (main toolbox process): bbox ordering, temporal smoothing of MHR pose parameters, optional **world-aligned** 3D keypoints/vertices from calibration, cross-view residual summary.

### Dense NPZ layout (same 5D idea as MMPose)

**Inside the GPU subprocess**, SAM writes a **raw** ``.npz`` with ``per_frame_outputs`` (Python/object arrays): a **list over frames** of per-camera bundles (dict or list, depending on ``mode``), each entry holding **per-person dicts** (``pred_keypoints_2d``, ``pred_keypoints_3d``, MHR parameters, etc.). That is **not** a single pre-allocated 5D array.

**After** ``Sam3dBody.post_inference``, the toolbox **materializes MMPose-style dense tensors** for consumers (see ``sam_3d_body_export_tensors.build_body_joints_npz_payload``). Typical shapes (subject × camera × frame × keypoint × channel):

| Key | Shape (conceptually) | axis4 |
|-----|----------------------|--------|
| ``2d``, ``2d_interpolated``, ``2d_filtered`` | ``(n_subjects, n_cameras, n_frames, n_keypoints, 3)`` | x, y, confidence |
| ``3d``, ``3d_camera`` (when present) | ``(n_subjects, n_cameras, n_frames, n_keypoints, 4)`` | x, y, z, confidence |
| ``3d_world`` slot (when present) | ``(n_subjects, 1, n_frames, n_keypoints, 4)`` | world coords + confidence |
| ``bbox_2d`` | ``(n_subjects, n_cameras, n_frames, 1, 5)`` | box + score |

Axis labels are documented in ``data_description`` inside each NPZ (``axis0``…``axis4``). Keypoint rows use **native MHR order** (``mhr_0`` … ``mhr_{K-1}``), not COCO row order; use ``[human_pose.sam3d_body_mhr]`` in ``predictions_mapping.toml`` for COCO-17 ↔ MHR index pairs.

**Mesh** NPZ (``body_mesh``): ``faces`` plus ``vertices`` (and optional ``vertices_world``) with their own ``data_description`` — not the same 5D keypoint tensor.

**Temporal smoothing** in post-process operates on **packed MHR parameter vectors** over time (not on these dense 5D keypoint arrays). The same Savitzky–Golay logic as MMPose paths is shared via ``adaptive_savgol_filter`` in ``method_detectors/filters.py``; ``SGFilter`` applies SciPy along the **frame** axis in one call per XY(Z) slab for classic ``(P, C, F, K, D)`` inputs.

- **Visualization** (when `visualize = true`): MHR skeleton edges on input frames under **`body_joints/sam_3d_body/visualization/<camera>/`**, plus per-camera MP4s. With **`visualize_mesh`** and **`save_vertices`**, mesh wireframes and MP4s go under **`body_mesh/sam_3d_body/visualization_3d/<camera>/`**. With **`visualize_mesh_interactive = true`**, Plotly HTML goes under **`body_mesh/sam_3d_body/visualization_3d_interactive/`**.

**COCO body-17 → MHR70 indices** (for comparisons and documentation; wrists differ from array order in MHR):

| COCO (name) | MHR index |
| - | - |
| nose | 0 |
| left_eye | 1 |
| right_eye | 2 |
| left_ear | 3 |
| right_ear | 4 |
| left_shoulder | 5 |
| right_shoulder | 6 |
| left_elbow | 7 |
| right_elbow | 8 |
| left_wrist | 62 |
| right_wrist | 41 |
| left_hip | 9 |
| right_hip | 10 |
| left_knee | 11 |
| right_knee | 12 |
| left_ankle | 13 |
| right_ankle | 14 |

The authoritative values (including visualization edges) live under **`[human_pose.sam_3d_body_mhr]`** in [`configs/predictions_mapping.toml`](../../configs/predictions_mapping.toml).


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

## WhisperX (Audio Transcription & Speaker Diarization)

**WhisperX** provides fast automatic speech recognition (ASR) with word-level timestamps and speaker diarization. By integrating voice activity detection (VAD) preprocessing and forced alignment, WhisperX significantly reduces hallucinations and improves timestamp accuracy compared to the original Whisper model. In the NICE toolbox, it is utilized to process audio tracks for transcription and to systematically identify who is speaking and when.  
[Bain et al., 2023](https://arxiv.org/abs/2303.00747)

### Voice Activity Detection (VAD)
The detection of the presence or absence of human speech. WhisperX uses VAD preprocessing to cleanly segment the audio. This crucial step enables efficient batched inference and reduces model hallucinations.

### Audio Transcription
Leverages the core Whisper architecture (via a `faster-whisper` backend) to generate highly accurate text predictions from the isolated speech segments.

### Audio Alignment
To achieve precise word-level timestamps, WhisperX applies "forced alignment". This process uses language-specific phoneme-based ASR models (such as `wav2vec2.0`) to align the orthographic transcriptions with the physical phonemes in the audio recording.

### Speaker Diarization
The process of partitioning the audio stream into homogeneous segments based on the identity of each speaker. WhisperX integrates `pyannote-audio` to cluster the speech segments, ultimately assigning discrete speaker ID labels to the timestamped words. *Note that currently we can not safely map the speaker labels to the actual subjects in the video.*

### Detector Configuration
The following parameters can be fine-tuned inside the detector configuration file:
- **`model_size`** (`"large-v3"`): The size of the underlying Whisper model. Larger models improve transcription accuracy but require more GPU memory.
- **`batch_size`** (`16`): The number of audio segments processed concurrently. Decrease this value if you encounter out-of-memory errors on smaller GPUs.
- **`language`** (`"en"`): Explicitly setting the language code speeds up processing and ensures the correct forced alignment model is queried.
- **`vad_onset` / `vad_offset`** (`0.9`): Activation and deactivation probability thresholds for generating speech segments in the VAD step.
- **`alignment_model_name`** (`"WAV2VEC2_ASR_LARGE_LV60K_960H"`): The specific Hugging Face repository name containing the phoneme-based model used to align transcriptions to the audio.
- **Hugging Face token for diarization**: Set `hugging_face_token` in **`machine_specific_paths.toml`** (gitignored). See [`effective_hf_hub_token`](../../nicetoolbox/utils/hf_token.py). There is no `hf_token` field under **`[algorithms.whisperx]`**; avoid committing secrets in **`detectors_config.toml`**.

