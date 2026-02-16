"""
GazeDistance feature detector class for 2 person gaze interaction components.
"""

import logging
import os

import numpy as np

from ....utils import linear_algebra as alg
from ..base_feature import BaseFeature
from ..gaze_interaction import utils as gaze_interaction_utils


class GazeDistance(BaseFeature):
    """
    The GazeDistance class is a feature detector that computes the gaze_interaction
    component.

    The GazeDistance feature detector accepts two primary inputs: the gaze_individual
    and face_landmarks components. These components are computed using the
    gaze_individual and body_joints method detectors, respectively. This feature
    detector calculates the smallest distance between a gaze direction vector and face
    landmarks within a 2-person context.
    Additionally, it has the ability to determine whether the gaze is directed
    at the face and if the gaze interaction is mutual.
    """

    components = ["gaze_interaction"]
    algorithm = "gaze_distance"

    def _get_input(self, component: str) -> np.ndarray:
        """
        Finds input file given a component.
        """
        for (comp, _algo), path in self.input_map.items():
            if comp == component:
                return np.load(path, allow_pickle=True)

        raise ValueError(f"Required input component '{component}' not found in input_map.")

    def compute(self):
        """
        This method computes the gaze_interaction component and saves the results as a
        compressed .npz file.

        It calculates the Euclidean distance between gaze direction vectors and face
        landmarks within a 2-person context. The distance is calculated between
        adjacent frames, measuring the change from t to t-1. The first frame will be
        empty.

        The method also determines whether the gaze is directed at the face (look_at)
        and if the gaze interaction is mutual.

        The results are saved as a compressed .npz file with the following structure:

        - distance_gaze: smallest distances from the gaze vector (of person A) to the
            face (of person B), and vice versa.
        - gaze_look_at: a boolean array indicating whether the gaze is directed at the
            face
        - gaze_mutual: a boolean array indicating whether the gaze is mutual
        - data_description: A dictionary containing the data description for all of the
            above output numpy arrays. See the documentation of the output for more
            details.

        Returns:
            visualization_data (list): A list containing the distances from the gaze to
            the face, a boolean array indicating whether the gaze is directed at the
            face, and a boolean array indicating whether the gaze is mutual.
        """
        keypoints_data = self._get_input("face_landmarks")
        if "3d" in keypoints_data:
            dim = "3d"
            dim_gaze = "gaze_fused_filtered"
        else:
            dim = "2d"
            dim_gaze = "gaze_2d_filtered"

        keypoints = keypoints_data[dim]
        keypoints_description = keypoints_data["data_description"].item()[dim]
        indices = ["face_landmarks" in key for key in keypoints_description["axis3"]]
        keypoint_values = keypoints[:, :, :, indices]
        head = keypoint_values[..., :-1].mean(axis=-2)

        gaze_data = self._get_input("gaze_multiview")
        gaze = gaze_data[dim_gaze]
        gaze_description = gaze_data["data_description"].item()[dim_gaze]

        assert gaze_description["axis0"] == keypoints_description["axis0"]
        subject_description = gaze_description["axis0"]
        if len(subject_description) < 2:
            logging.info(
                f"The selected data shows {len(subject_description)} subjects. "
                "Gaze interaction can not be calculated."
            )
            return None

        distance_p1 = alg.distance_line_point(head[0], gaze[0], head[1])
        distance_p2 = alg.distance_line_point(head[1], gaze[1], head[0])
        distances_face = np.stack((distance_p1, distance_p2), axis=0)

        # calculate look_at and mutual_gaze
        look_at = distances_face <= self.detector_config.threshold_look_at
        mutual = np.all(look_at, axis=0, keepdims=True)
        visualization_data = [distances_face, look_at, mutual]

        # reshape arrays
        def reshape(arr):
            return np.stack(
                (
                    np.concatenate((np.zeros_like(arr[0]), arr[0]), axis=-1),
                    np.concatenate((arr[1], np.zeros_like(arr[1])), axis=-1),
                ),
                axis=0,
            )

        distances_face = reshape(distances_face)
        look_at = reshape(look_at)
        mutual = reshape(np.concatenate((mutual, mutual), axis=0))

        #  save as npz file
        data_description = dict(axis0=subject_description, axis1=[dim], axis2=gaze_description["axis2"])
        out_dict = {
            f"distance_gaze_{dim}": distances_face,
            f"gaze_look_at_{dim}": look_at,
            f"gaze_mutual_{dim}": mutual,
            "data_description": {
                f"distance_gaze_{dim}": dict(
                    **data_description,
                    axis3=[f"to_face_{subj}" for subj in subject_description],
                ),
                f"gaze_look_at_{dim}": dict(
                    **data_description,
                    axis3=[f"look_at_{subj}" for subj in subject_description],
                ),
                f"gaze_mutual_{dim}": dict(
                    **data_description,
                    axis3=[f"with_{subj}" for subj in subject_description],
                ),
            },
        }
        filepath = os.path.join(self.result_folders["gaze_interaction"], f"{self.algorithm}.npz")
        np.savez_compressed(filepath, **out_dict)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return visualization_data

    def visualization(self, data):
        """
        Creates visualizations for the computed gaze interaction features.

        This method generates line graphs showing the distance between
        gaze points and face landmarks, binary graphs indicating whether
        the gaze is directed at the face, and binary graphs indicating
        whether the gaze is mutual. Additionally, it creates videos of
        these line graphs evolving over time.

        Args:
            data (tuple): Output data from the compute method containing:
                - the distances of the gaze to the face
                - a boolean array indicating whether the gaze is directed at the face
                - a boolean array indicating whether the gaze is mutual.

        Returns:
            None
        """
        if data is not None:
            logging.info(
                f"Visualizing the feature detector output {self.components}."
                f"This may take longer due to the evolving linegraph video creation."
            )

            distances, look_at, mutual = data
            mutual = np.concatenate((mutual, mutual), axis=0)

            # Determine global_min and global_max - define y-lims of graphs
            # global_min = min(distances[0].min(), distances[1].min())
            # global_max = max(distances[0].max(), distances[1].max())

            # scale binary data
            # look_at = look_at * (global_max - global_min) / 2
            # look_at += global_min
            # mutual = mutual * (global_max - global_min) / 2
            # mutual += global_min

            input_data = np.concatenate((distances, look_at, mutual), axis=-1)
            categories = ["distance_gaze_face", "gaze_look_at", "gaze_mutual"]
            gaze_interaction_utils.visualize_gaze_interaction(
                input_data, categories, self.viz_folder, self.subjects_descr
            )
            logging.info(f"Visualization of feature detector {self.components} completed.")
