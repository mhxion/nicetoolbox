"""
MMPose 2D frameworks and specific algorithm definitions.
"""

import logging
import os

import numpy as np

from ....utils import check_and_exception as check
from ....utils import triangulation as tri
from ... import config_handler as confh
from ..filters import SGFilter
from . import pose_utils
from .mmpose_framework_base import BaseMMPose, extract_key_per_value


class MMPose2D(BaseMMPose):
    """
    The MMPose2D class is a method detector for 2D pose estimation using the MMPose
    framework.

    Its post_inference method computes 3D points via multi-view stereo triangulation.
    """

    def _setup_subprocess_settings(self) -> None:
        super()._setup_subprocess_settings()
        self.script_path = self.io.get_inference_path("body_joints", "mmpose_2d")  # TODO: fix me

    def post_inference(self):
        """
        Post-inference processing for pose estimation components such as body_joints,
        hand_joints, and face_landmarks.

        This method takes the raw 2D pose estimation results and applies a series of
        processing steps. They include optional filtering to smooth the results,
        interpolation to fill in missing values, undistortion using camera calibration
        parameters, and 3D triangulation from multiple camera views. The final processed
        results are saved for further analysis and visualization for each of the
        components.

        Steps:
        1. Filtering: Applies a smoothing filter to the 2D pose estimation results
            if filtering is enabled. This step reduces noise and improves the
            consistency of the pose data over time.
        2. Interpolation: Fills in missing values in the 2D pose estimation results.
            This is crucial for maintaining the integrity of the pose data, especially
            in cases where occlusion or poor lighting conditions may lead to incomplete
            detections.
        3. Undistortion: Corrects the 2D pose estimation results for lens distortion
            using the camera's calibration parameters.
        4. 3D Triangulation: Uses the undistorted 2D pose estimation results from
            at least two camera views to reconstruct the 3D positions of the pose
            keypoints.
        5. Saving Results: The processed 3D pose data is saved to a .npz file with
            the following structure:
                - '2d': A numpy array containing the 2D pose estimation results.
                - '2d_filtered': A numpy array containing the filtered 2D pose
                    estimation results.
                - '2d_interpolated': A numpy array containing the interpolated 2D pose
                    estimation results.
                - 'bbox_2d': A numpy array containing the 2D bounding box coordinates.
                - '3d': A numpy array containing the 3D pose estimation results.
                - 'data_description': A dictionary containing the data description for
                    the above output numpy arrays. See the documentation of the output
                    for more details.

        Returns:
            None. The processed results are saved to the output folder (See step 5).
        """
        for _component, result_folder in self.result_folders.items():
            prediction_file = os.path.join(result_folder, f"{self.algorithm}.npz")
            prediction = np.load(prediction_file, allow_pickle=True)
            data_description = prediction["data_description"].item()
            results_2d = prediction["2d"]
            results_2d_bbox = prediction["bbox_2d"]

            iou_array = pose_utils.create_iou_all_pairs(results_2d_bbox)

            # Filter is applied.
            if self.filtered:
                logging.info("APPLYING filtering to 3d data...")
                results_2d_filtered = results_2d.copy()
                filter = SGFilter(self.filter_window_length, self.filter_polyorder)
                results_2d_filtered = filter.apply(results_2d_filtered)

            # If the filter is applied, 2d interpolation is performed on filtered data.
            if self.filtered:
                results_2d_interpolated = results_2d_filtered.copy()
            else:
                results_2d_interpolated = results_2d.copy()

            keypoint_conf_threshold = self.min_detection_confidence

            # A mask is created where the confidence score is below the threshold.
            low_confidence_mask = results_2d_interpolated[:, :, :, :, 2] < keypoint_conf_threshold

            # The mask is applied to set the first and second values of num_estimates to NaN
            # where the confidence is low.
            results_2d_interpolated[low_confidence_mask, 0:2] = np.nan
            results_2d_interpolated = pose_utils.interpolate_data(results_2d_interpolated, is_3d=False)

            data_description["2d_interpolated"] = data_description["2d"]
            data_description.update(
                {
                    "bbox_overlap": dict(
                        axis0=data_description["2d"]["axis0"],
                        axis1=data_description["2d"]["axis1"],
                        axis2=data_description["2d"]["axis2"],
                        axis3=[f"with_{subj}" for subj in self.subjects_descr],
                    )
                }
            )
            can_estimate_3d = len(self.camera_names) >= 2
            if can_estimate_3d and not self.calibration:
                logging.warning(
                    "WARNING - Calibration file is not valid. "
                    "Therefore, cannot compute 3d positions of the joints."
                    "Please see docs/wikis/wiki_calibration.md"
                )
                can_estimate_3d = False

            if not can_estimate_3d:
                if self.filtered:
                    data_description["2d_filtered"] = data_description["2d"]
                    # Results are saved.
                    results_dict = {
                        "2d": results_2d,
                        "2d_filtered": results_2d_filtered,
                        "2d_interpolated": results_2d_interpolated,
                        "bbox_2d": results_2d_bbox,
                        "bbox_overlap": iou_array,
                        "data_description": data_description,
                    }
                    np.savez_compressed(prediction_file, **results_dict)

                else:
                    # Results are saved.
                    results_dict = {
                        "2d": results_2d,
                        "2d_interpolated": results_2d_interpolated,
                        "bbox_2d": results_2d_bbox,
                        "bbox_overlap": iou_array,
                        "data_description": data_description,
                    }
                    np.savez_compressed(prediction_file, **results_dict)

            else:
                logging.info("COMPUTING 3d position of the joints...")

                if len(self.camera_names) > 2:
                    logging.warning(
                        f"WARNING - The 2D positions of the joints have been estimated "
                        "for more than two cameras. \n"
                        f"The 3D positions will be computed using the first two "
                        "cameras specified in the camera_names parameter in the "
                        "detectors_config.toml file \n"
                        f"{self.camera_names[0]} & {self.camera_names[1]}"
                    )

                # Interpolated_2d results are used instead of original 2d.
                cam1_data, cam2_data = (
                    results_2d_interpolated[:, 0],
                    results_2d_interpolated[:, 1],
                )

                # Subject indices common in both camera views are found.
                subjects_cam1 = set(self.cam_sees_subjects[self.camera_names[0]])
                subjects_cam2 = set(self.cam_sees_subjects[self.camera_names[1]])
                common_subjects_idx = list(subjects_cam1 & subjects_cam2)

                person_data_list = []
                for subject_idx in common_subjects_idx:
                    person_cam1 = cam1_data[subject_idx]
                    person_cam2 = cam2_data[subject_idx]

                    # The x and y values are extracted.
                    xy_points_cam1 = person_cam1[:, :, :2].reshape(-1, 1, 2)
                    xy_points_cam2 = person_cam2[:, :, :2].reshape(-1, 1, 2)

                    # Confidence scores are extracted.
                    conf_cam1 = person_cam1[:, :, 2].reshape(-1, 1, 1)
                    conf_cam2 = person_cam2[:, :, 2].reshape(-1, 1, 1)

                    # Since interpolated data is used, some missing values might be present.
                    # A combined mask for NaN values in either camera's data is created.
                    nan_mask_cam1 = np.isnan(xy_points_cam1).any(axis=2)
                    nan_mask_cam2 = np.isnan(xy_points_cam2).any(axis=2)
                    combined_nan_mask = nan_mask_cam1 | nan_mask_cam2

                    # Rows with NaNs are filtered out for processing.
                    filtered_xy_points_cam1 = xy_points_cam1[~combined_nan_mask]
                    filtered_xy_points_cam2 = xy_points_cam2[~combined_nan_mask]

                    filtered_confidence_cam1 = conf_cam1[~combined_nan_mask]
                    filtered_confidence_cam2 = conf_cam2[~combined_nan_mask]

                    # Data is undistorted.
                    cam1_undistorted = np.squeeze(
                        tri.undistort_points_pinhole(
                            filtered_xy_points_cam1,
                            np.array(self.calibration[self.camera_names[0]]["intrinsic_matrix"]),
                            np.array(self.calibration[self.camera_names[0]]["distortions"]),
                        )
                    )
                    cam2_undistorted = np.squeeze(
                        tri.undistort_points_pinhole(
                            filtered_xy_points_cam2,
                            np.array(self.calibration[self.camera_names[1]]["intrinsic_matrix"]),
                            np.array(self.calibration[self.camera_names[1]]["distortions"]),
                        )
                    )

                    # Data is triangulated.
                    person_data_3d = tri.triangulate_stereo(
                        np.array(self.calibration[self.camera_names[0]]["projection_matrix"]),
                        np.array(self.calibration[self.camera_names[1]]["projection_matrix"]),
                        cam1_undistorted.T,
                        cam2_undistorted.T,
                    )

                    # Confidence score is added; cam1 and cam2 are combined, and the minimum confidence value is kept.
                    confidence_combined = np.minimum(filtered_confidence_cam1, filtered_confidence_cam2)
                    person_data_3d_with_conf = np.concatenate([person_data_3d.T, confidence_combined], axis=1)

                    # 3d array is reshaped.
                    # Output arrays filled with NaNs are created.
                    output_shape = (xy_points_cam1.shape[0], 4)
                    output_data_3d = np.full(output_shape, np.nan)

                    # The processed data is inserted back into the correct positions.
                    output_data_3d[~combined_nan_mask.reshape(-1)] = person_data_3d_with_conf
                    reshaped_3D_points = output_data_3d.reshape(person_cam1.shape[0], person_cam1.shape[1], 4)
                    person_data_list.append(reshaped_3D_points)

                # Any [0,0,0] prediction is checked.
                for person_data in person_data_list:
                    check.check_zeros(person_data)

                # Results are saved.
                descr_2d = data_description["2d"]
                common_subjects = [s for i, s in enumerate(self.subjects_descr) if i in common_subjects_idx]
                data_description.update(
                    {
                        "3d": dict(
                            axis0=common_subjects,
                            axis1=["3d"],
                            axis2=descr_2d["axis2"],
                            axis3=descr_2d["axis3"],
                            axis4=[
                                "coordinate_x",
                                "coordinate_y",
                                "coordinate_z",
                                "confidence_score",
                            ],
                        )
                    }
                )
                if self.filtered:
                    data_description["2d_filtered"] = data_description["2d"]
                    results_dict = {
                        "2d": results_2d,
                        "2d_filtered": results_2d_filtered,
                        "2d_interpolated": results_2d_interpolated,
                        "bbox_2d": results_2d_bbox,
                        "bbox_overlap": iou_array,
                        "3d": np.stack(person_data_list)[:, None],
                        "data_description": data_description,
                    }
                    np.savez_compressed(prediction_file, **results_dict)
                else:
                    results_dict = {
                        "2d": results_2d,
                        "2d_interpolated": results_2d_interpolated,
                        "bbox_2d": results_2d_bbox,
                        "bbox_overlap": iou_array,
                        "3d": np.stack(person_data_list)[:, None],
                        "data_description": data_description,
                    }
                    np.savez_compressed(prediction_file, **results_dict)


# =============================================================================
# 2D Algorithm Implementations
# =============================================================================


class HRNetw48(MMPose2D):
    """
    HRNetw48 is a subclass of MMPose specialized for pose estimation using the HRNetw48
    model.

    Components: body_joints, hand_joints, face_landmarks
    """

    components = ["body_joints", "hand_joints", "face_landmarks"]
    algorithm = "hrnetw48"

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        """
        Extracts and returns the indices and descriptions of keypoints for each
        component.

        Args:
            keypoints_indices (dict): A dictionary containing the indices of keypoints
                for each component. The keys of the dictionary are the component names
                ('body_joints', 'hand_joints', 'face_landmarks'), and the values are
                dictionaries containing the indices of keypoints for each keypoint.

        Returns:
            tuple: A tuple containing two dictionaries.
                - The first dictionary contains the indices of keypoints for each
                    component.
                - The second dictionary contains the descriptions of keypoints for
                    each component.
        """
        indices = dict(
            body_joints=confh.flatten_list(
                list(keypoints_indices.body.values()) + list(keypoints_indices.foot.values())
            ),
            hand_joints=confh.flatten_list(list(keypoints_indices.hand.values())),
            face_landmarks=confh.flatten_list(list(keypoints_indices.face.values())),
        )
        description = dict(
            body_joints=confh.flatten_list(
                extract_key_per_value(keypoints_indices.body) + extract_key_per_value(keypoints_indices.foot)
            ),
            hand_joints=confh.flatten_list(extract_key_per_value(keypoints_indices.hand)),
            face_landmarks=confh.flatten_list(extract_key_per_value(keypoints_indices.face)),
        )
        return indices, description


class VitPose(MMPose2D):
    """
    VitPose is a subclass of MMPose specialized for pose estimation using the Vision
    Transformer (ViT) model.

    Component: body_joints
    """

    components = ["body_joints"]
    algorithm = "vitpose"

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        """
        Extracts and returns the indices and descriptions of keypoints for each
        component.

        Args:
            keypoints_indices (dict): A dictionary containing the indices of keypoints
                for each component. The keys of the dictionary are the component names
                ('body_joints', 'hand_joints', 'face_landmarks'), and the values are
                dictionaries containing the indices of keypoints for each keypoint.
                Note: This algorithm only supports the 'body_joints' component.

        Returns:
            tuple: A tuple containing two dictionaries.
                - The first dictionary contains the indices of keypoints for each
                    component.
                - The second dictionary contains the descriptions of keypoints for each
                    component.

        """
        indices = dict(body_joints=confh.flatten_list(list(keypoints_indices.body.values())))
        description = dict(body_joints=confh.flatten_list(extract_key_per_value(keypoints_indices.body)))
        return indices, description


class VitPoseHuge(MMPose2D):
    """
    VitPoseHuge is a subclass for the Huge variant of ViT.
    """

    algorithm = "vitpose_huge"
    components = ["body_joints"]

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        indices = dict(body_joints=confh.flatten_list(list(keypoints_indices.body.values())))
        description = dict(body_joints=confh.flatten_list(extract_key_per_value(keypoints_indices.body)))
        return indices, description


class RTMPoseLAIC(MMPose2D):
    """
    RTMPoseLAIC is a subclass of MMPose for the RTMPose-L model.
    (Pre-trained on AIC, fine-tuned on COCO: Outputs 17 Body Keypoints).
    """

    algorithm = "rtmpose_l_aic"
    components = ["body_joints"]

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        indices = dict(body_joints=confh.flatten_list(list(keypoints_indices.body.values())))
        description = dict(body_joints=confh.flatten_list(extract_key_per_value(keypoints_indices.body)))
        return indices, description


class RTMPoseWHolebody(MMPose2D):
    r"""
    Handler for RTMPose models trained on COCO-Wholebody \(133 keypoints).
    Outputs in one pass: Body, Feer, Face, and Hands.
    """

    algorithm = "rtmpose_l_wholebody"
    components = ["body_joints", "hand_joints", "face_landmarks"]

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        def extract_keys(d):
            if all(isinstance(v, int) for v in d.values()):
                return list(d.keys())
            res = []
            for k, v in d.items():
                if isinstance(v, list):
                    res.extend([f"{k}_{i}" for i in range(len(v))])
                elif isinstance(v, int):
                    res.append(v)
            return res

        indices = dict(
            body_joints=confh.flatten_list(
                list(keypoints_indices.body.values()) + list(keypoints_indices.foot.values())
            ),
            hand_joints=confh.flatten_list(list(keypoints_indices.hand.values())),
            face_landmarks=confh.flatten_list(list(keypoints_indices.face.values())),
        )

        description = dict(
            body_joints=confh.flatten_list(extract_keys(keypoints_indices.body) + extract_keys(keypoints_indices.foot)),
            hand_joints=confh.flatten_list(extract_keys(keypoints_indices.hand)),
            face_landmarks=confh.flatten_list(extract_keys(keypoints_indices.face)),
        )
        return indices, description


class RTMPoseMPII(MMPose2D):
    """
    Handler for RTMPose models trained on MPII (16 Keypoints).
    """

    algorithm = "rtmpose_m_mpii"
    components = ["body_joints"]

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        indices = dict(body_joints=confh.flatten_list(list(keypoints_indices.body.values())))
        description = dict(body_joints=confh.flatten_list(extract_key_per_value(keypoints_indices.body)))
        return indices, description
