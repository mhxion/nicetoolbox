# Changelog

## Unreleased

- **MotionBERT** moved from component **`body_joints_lifted`** to **`body_joints_local`** (output still **`body_joints_local/motionbert.npz`**). Update **`[component_algorithm_mapping]`** and any evaluation blocks that referenced **`body_joints_lifted`**.
- **SAM 3D Body** (`sam_3d_body`): processed outputs are under **`…/<component>/sam_3d_body/`** (next to **`run_config`**). Raw inference is **`_sam_3d_body_inference_raw.npz`** (not under **`detector_output/`**). **`body_joints_local`** always gets **`sam_3d_body_camera.npz`** (camera-native **`3d`**). **`body_joints`** gets **`sam_3d_body.npz`** when every configured camera has **intrinsic + projection** in the sequence calibration. With **multi-view** and **`stereo_triangulation_body_joints`** (default **true**), primary **`3d`** uses **two-view** **`cv2.triangulatePoints`** on undistorted SAM **2d** keypoints from the **first two** **`camera_names`** (same convention as **vitpose_huge**; not N-view fusion). When stereo is disabled or not applicable, primary **`3d`** remains world-broadcast from calibration alignment when world coords exist. **`3d_world`** / **`3d_camera`** are still written when available. Without calibration, the **`body_joints`** NPZ is skipped. **`data_description.sam_3d_body.export_policy`** documents the policy. Optional JSON/CSV raw dumps go under **`body_joints_local/sam_3d_body/detector_output/`**. 2D skeleton visualization is written under **`body_joints/sam_3d_body/visualization/`** and duplicated under **`body_joints_local/sam_3d_body/visualization/`**; mesh wireframes only under **`body_mesh/sam_3d_body/visualization_3d/`**.
- **predictions_mapping**: optional schema **`[human_pose.sam_3d_body_mhr_evaluation]`** (same fields as **`sam_3d_body_mhr`**) for evaluation-specific MHR mapping.
- **WhisperX**: **`hf_weights_cache_dir`** in **`detectors_config.toml`** (default **`<assets>/whisperx`**) is the Hugging Face / WhisperX cache directory (created at inference). No fake **`asset_manifest.toml`** entry is required.

- **MotionBERT** (`motionbert`): MMPose-based **3D body pose lifting** from a precomputed **`body_joints`** NPZ (default 2D source **`vitpose_huge`**). Lifting uses stored 2D keypoints/boxes from the NICE 2D pipeline, not a second 2D inferencer pass; optional **`Pose3DInferencer`** is for visualization only. Post-processing keeps **root-relative** `3d` (PnP / `2d_projected` / `3d_pnp_world` removed).

## 0.2.2
- Refactoring of data preprocessing and inference for all detectors.
- Major optimization and bug-fixing of py-feat inference.
- Refactoring, optimization, and bug-fixing of multiview-ethgaze.
- Refactoring of config placeholders resolution, making it faster and more stable.
- New config validation system. It will detect missing required fields or wrong field types across all configs.
- Fixes for subject tracking consistency in multiple detectors.
- In `detectors_run_file.toml` you can set `video_length = -1` to process all frames inside a video.

**Breaking changes:**
- The frame index leading zeroes format was extended from `05d` to `09d` to support longer videos. This results in new filenames.
- CSV exported files are now saved inside individual video folders, not inside the root output folder. This can be customized in config.
- All runtime placeholders now start with `cur_<placeholder_name>`. For example, the `<session_ID>` placeholder was renamed to `<cur_session_ID>`.
- Cyclic placeholder dependencies are deprecated. For example, `git_hash = "<git_hash>"` will now raise an error.
- Placeholder shadowing is deprecated. Use unique placeholder names at each level of the config file.
- NICE Toolbox now uses submodule forks of [mmpose](https://github.com/OSLabTools/mmpose) and [SPIGA](https://github.com/OSLabTools/SPIGA). Library versions remain the same, so there should be no changes in results.
- [Multiview-ETH-XGaze](https://github.com/OSLabTools/ETH_XGaze) now supports multiview only inside NICE Toolbox. All logic for multi-camera fusion was moved to NICE.
- `eth_xgaze` now exports raw `3d` and `3d_filtered` for individual cameras and `xgaze_gaze_fused` and `xgaze_gaze_fused_filtered` fused from all cameras.
- `eth_xgaze` now exports `landmarks_2d` with confidence scores.
- `detectors_run_file.toml` config now requires `log_level` and `error_level` fields to be set.

## 0.2.1

- Evaluation module, Docker support, additional detector output, and many other improvements.

## 0.2.0

- Code refactoring, easier installation, and new detectors for emotion individuals and head orientation.

## 0.1.0

- Initial release.