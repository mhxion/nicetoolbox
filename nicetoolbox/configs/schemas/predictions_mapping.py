from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, computed_field, field_validator


class CocoWholebodyKeypointsIndex(BaseModel):
    body: Dict[str, int]
    foot: Dict[str, int]
    face: Dict[str, List[int]]
    hand: Dict[str, List[int]]


class CocoWholebodyBodypartIndex(BaseModel):
    head: List[int]
    upper_body: List[int]
    lower_body: List[int]


class CocoWholebodyConnections(BaseModel):
    body_joints: List[List[str]]
    hand_joints: List[List[str]]
    face_landmarks: List


class CocoWholebody(BaseModel):
    keypoints_index: CocoWholebodyKeypointsIndex
    bodypart_index: CocoWholebodyBodypartIndex
    connections: CocoWholebodyConnections


class Human36mKeypointsIndex(BaseModel):
    body: Dict[str, int]
    foot: Dict[str, int]
    face: Dict[str, List[int]]
    hand: Dict[str, List[int]]


class Human36mConnections(BaseModel):
    body_joints: List[List[str]]
    hand_joints: List
    face_landmarks: List


class Human36m(BaseModel):
    keypoints_index: Human36mKeypointsIndex
    connections: Human36mConnections


class MpiiKeypointsIndex(BaseModel):
    body: Dict[str, int]


class MpiiConnections(BaseModel):
    body_joints: List[List[str]]


class Mpii(BaseModel):
    keypoints_index: MpiiKeypointsIndex
    connections: MpiiConnections


class Sam3dBodyMhrKeypointsIndex(BaseModel):
    """Minimal ``body`` index so proximity / feature detectors can resolve ``nose`` etc. on MHR rows."""

    body: Dict[str, int]


class Sam3dBodyMhrBodypartIndex(BaseModel):
    """MHR vertex indices grouped like coco_wholebody bodyparts (for kinematics viz)."""

    head: List[int]
    upper_body: List[int]
    lower_body: List[int]


class Sam3dBodyMhr(BaseModel):
    """
    COCO body-17 ↔ MHR70 for SAM 3D Body: 2D skeleton overlays and upstream-facing helpers.

    ``coco_body_17_joint_names`` and ``coco_body_17_mhr70_index`` are parallel (COCO order 0..16).
    ``keypoints_index`` / ``bodypart_index`` are derived for tooling that expects coco_wholebody-like shapes.
    """

    coco_body_17_joint_names: List[str] = Field(..., min_length=17, max_length=17)
    coco_body_17_mhr70_index: List[int] = Field(..., min_length=17, max_length=17)
    mhr_body_vis_edges: List[Tuple[int, int]] = Field(default_factory=list)

    @computed_field
    @property
    def keypoints_index(self) -> Sam3dBodyMhrKeypointsIndex:
        body = dict(zip(self.coco_body_17_joint_names, self.coco_body_17_mhr70_index))
        return Sam3dBodyMhrKeypointsIndex(body=body)

    @computed_field
    @property
    def bodypart_index(self) -> Sam3dBodyMhrBodypartIndex:
        mhr = self.coco_body_17_mhr70_index
        head = [mhr[i] for i in range(5)]
        upper_body = [mhr[i] for i in range(5, 11)]
        lower_body = [mhr[i] for i in range(11, 17)]
        return Sam3dBodyMhrBodypartIndex(head=head, upper_body=upper_body, lower_body=lower_body)

    @field_validator("mhr_body_vis_edges", mode="before")
    @classmethod
    def _coerce_vis_edges(cls, v: Any) -> List[Tuple[int, int]]:
        if v is None:
            return []
        out: List[Tuple[int, int]] = []
        for edge in v:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                out.append((int(edge[0]), int(edge[1])))
            else:
                raise ValueError(f"mhr_body_vis_edges entries must be length-2 lists, got {edge!r}")
        return out


class Sam3dBodyMhrEvaluation(Sam3dBodyMhr):
    """Optional evaluation-time MHR↔semantic mapping; duplicate or specialize vs sam_3d_body_mhr in TOML."""

    pass


class HumanPose(BaseModel):
    coco_wholebody: CocoWholebody
    human36m: Human36m
    mpii: Mpii
    sam_3d_body_mhr: Sam3dBodyMhr
    sam_3d_body_mhr_evaluation: Optional[Sam3dBodyMhrEvaluation] = None
    bodypart_names: Dict[str, List[str]]
    bone_dict: Dict[str, List[str]]
    # Same bone names as bone_dict; joint keys are mhr_<i> for SAM 3D Body NPZ axis3.
    bone_dict_mhr: Dict[str, List[str]]
    joint_diameter_size: Dict[str, float]


class SpigaKeypointsIndex(BaseModel):
    face: Dict[str, List[int]]


class Spiga(BaseModel):
    keypoints_index: SpigaKeypointsIndex


class HeadOrientation(BaseModel):
    spiga: Spiga


class Order(BaseModel):
    keypoints: List[str]
    bones: List[str]
    microactions: List[str]


class Microactions(BaseModel):
    daya: Dict[str, int]


# Top-level keys in predictions_mapping.toml
class PredictionsMappingConfig(BaseModel):
    human_pose: HumanPose
    head_orientation: HeadOrientation
    order: Order
    microactions: Microactions
