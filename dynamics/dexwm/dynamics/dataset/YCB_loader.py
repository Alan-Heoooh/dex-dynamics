# this dataloader loads 500 out of 1000, only for the right hand

"""DexYCB dataset."""

from pathlib import Path
import json
import numpy as np
import yaml
from typing import Literal
import os
from .ycb_utils import find_longest_true_sequence


_SUBJECTS = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
    # "20201022-subject-10-modified",
]
invalid_lst = [
    "20200820-subject-03+20200820_143206",
    "20201002-subject-08+20201002_111616",
    "20201022-subject-10+20201022_113502",
    "20200908-subject-05+20200908_143832",
    "20200908-subject-05+20200908_145430",
    "20200928-subject-07+20200928_145424",
    "20201002-subject-08+20201002_110425",
    "20201015-subject-09+20201015_143338",
    "20201015-subject-09+20201015_144651",
    "20201015-subject-09+20201015_143338",
    "20200928-subject-07+20200928_145204",
]
# invalid_lst = [
#     "20200820-subject-03+20200820_143206+839512060362",
#     "20200820-subject-03+20200820_143206+840412060917",
#     "20200820-subject-03+20200820_143206+932122061900",
#     "20201002-subject-08+20201002_111616+841412060263",
#     "20201002-subject-08+20201002_111616+839512060362",
#     "20201002-subject-08+20201002_111616+840412060917",
#     "20201022-subject-10+20201022_113502+839512060362",
#     "20200820-subject-03+20200820_141302+841412060263",
#     "20200820-subject-03+20200820_141302+840412060917",
#     "20200908-subject-05+20200908_143832+839512060362",
#     "20200908-subject-05+20200908_143832+932122060857",
#     "20200908-subject-05+20200908_145430+932122062010",
#     "20200928-subject-07+20200928_145424+836212060125",
#     "20201002-subject-08+20201002_110425+841412060263",
#     "20201015-subject-09+20201015_143338+841412060263",
#     "20201015-subject-09+20201015_144651+841412060263",
#     "20201015-subject-09+20201015_143338+932122062010",
#     "20201015-subject-09+20201015_143338+932122060861",
#     "20201015-subject-09+20201015_143338+839512060362",
#     "20200928-subject-07+20200928_145204+836212060125",
# ]


YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

_MANO_JOINTS = [
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]

_MANO_JOINT_CONNECT = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]

_SERIALS = [
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4


class DexYCBVideoDataset:
    def __init__(
        self,
        data_dir,
        hand_type,
        filter_objects=[],
        file_path = None,
        mode: Literal["train", "test"] = "train",
    ):
        self._data_dir = Path(data_dir)
        self._calib_dir = self._data_dir / "calibration"
        self._model_dir = self._data_dir / "models"
        self._split_dir = self._data_dir / "annotations" / "annotation_folder"

        if file_path is None:
            """Not specified file path, load all possible data"""
            if not os.path.exists(self._data_dir / f"dex_ycb_split_{mode}_data.json"):
                with open(self._split_dir / f"dex_ycb_s0_{mode}_data.json", "r") as f:
                    self._data_split = json.load(f)

                clean_split = {}
                for key, value in self._data_split.items():
                    sequence_folder = value["color_file"].split("/")[0]
                    sequence_dir = value["color_file"].split("/")[1]
                    if sequence_folder not in clean_split:
                        clean_split[sequence_folder] = set()
                    if f"{sequence_folder}+{sequence_dir}" in invalid_lst:
                        continue
                    clean_split[sequence_folder].add(sequence_dir)
                for sequence_folder in clean_split:
                    clean_split[sequence_folder] = list(clean_split[sequence_folder])
                output_path = self._data_dir / f"dex_ycb_split_{mode}_data.json"
                with open(output_path, "w") as json_file:
                    json.dump(clean_split, json_file, indent=4)
            else:
                with open(self._data_dir / f"dex_ycb_split_{mode}_data.json", "r") as f:
                    clean_split = json.load(f)
        
        # print(self._)
        # Filter
        self.use_filter = True  # len(filter_objects) > 0
        if len(filter_objects) > 0:
            inverse_ycb_class = {
                "_".join(value.split("_")[1:]): key
                for key, value in YCB_CLASSES.items()
            }
            ycb_object_names = list(inverse_ycb_class.keys())
            filter_ids = []
            for obj in filter_objects:
                if obj not in ycb_object_names:
                    print(f"Filter object name {obj} is not a valid YCB name")
                else:
                    filter_ids.append(inverse_ycb_class[obj])
        else:
            filter_ids = [i for i in range(1, 22)]

        # assert len(filter_ids) == 1, "Only support one object filter"

        # Camera and mano
        self._intrinsics, self._extrinsics = self._load_camera_parameters()
        self._mano_side = hand_type
        self._mano_parameters = self._load_mano()
        print(hand_type)

        

        self._capture_meta = {}
        self._capture_pose = {}
        self._capture_filter = {}
        self._captures = []

        if file_path is None:
            # Capture data
            self._subject_dirs = [
                sub for sub in self._data_dir.iterdir() if sub.stem in _SUBJECTS
            ]
        else:
            self._subject_dirs = [file_path.split("/")[0]]
            
        for subject_dir in self._subject_dirs:
            if file_path is None:
                capture_dirs = []
                for capture_dir in subject_dir.iterdir():
                    if (
                        subject_dir.stem not in clean_split
                        or capture_dir.stem not in clean_split[subject_dir.stem]
                    ):
                        continue
                    capture_dirs.append(capture_dir)
                    
            else:
                capture_dirs = [Path(os.path.join(self._data_dir, file_path))]
                
            for capture_dir in capture_dirs:
                
                # meta_file = os.path.join(capture_dir, "meta.yml")
                meta_file = capture_dir / "meta.yml"
                with meta_file.open(mode="r") as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)

                if hand_type not in meta["mano_sides"]:
                    continue

                pose = np.load((capture_dir / "pose.npz").resolve().__str__())
                if self.use_filter:
                    ycb_ids = meta["ycb_ids"]
                    # Skip current capture if no desired object inside
                    if (
                        len(
                            list(
                                set([ycb_ids[meta["ycb_grasp_ind"]]]) & set(filter_ids)
                            )
                        )
                        < 1
                    ):
                        continue
                    capture_filter = [
                        i for i in range(len(ycb_ids)) if ycb_ids[i] in filter_ids
                    ]
                    object_pose = pose["pose_y"]
                    frame_indices, filter_id = self._filter_object_motion_frame(
                        capture_filter, object_pose
                    )
                    frame_indices = self._filter_occluded_frame(
                        frame_indices, pose["pose_m"]
                    )
                    if len(frame_indices) < 20 or (
                        len(frame_indices) < 62 and mode == "test"
                    ):
                        continue
                    self._capture_filter[capture_dir.stem] = [filter_id]
                self._capture_meta[capture_dir.stem] = meta
                self._capture_pose[capture_dir.stem] = pose
                self._captures.append(capture_dir.stem)
                # print(f"Loaded capture {capture_dir.stem}")

    def __len__(self):
        return len(self._captures)

    def __getitem__(self, item):
        if item > self.__len__():
            raise ValueError(f"Index {item} out of range")

        capture_name = self._captures[item]
        meta = self._capture_meta[capture_name]
        pose = self._capture_pose[capture_name]
        hand_pose = pose["pose_m"]
        object_pose = pose["pose_y"]
        ycb_ids = meta["ycb_ids"]
        mano_side = meta["mano_sides"][0]

        # Load extrinsic and mano parameters
        extrinsic_name = meta["extrinsics"]  # this is camera extrinsics
        extrinsic_mat = np.array(
            self._extrinsics[extrinsic_name]["extrinsics"]["apriltag"]
        ).reshape([3, 4])
        extrinsic_mat = np.concatenate(
            [extrinsic_mat, np.array([[0, 0, 0, 1]])], axis=0
        )
        mano_name = meta["mano_calib"][0]
        mano_parameters = self._mano_parameters[mano_name]

        if self.use_filter:
            capture_filter = np.array(self._capture_filter[capture_name])
            frame_indices, _ = self._filter_object_motion_frame(
                capture_filter, object_pose
            )
            # ycb_ids = [
            #     ycb_ids[valid_id] for valid_id in self._capture_filter[capture_name]
            # ]

            frame_indices = self._filter_occluded_frame(frame_indices, pose["pose_m"])

            hand_pose = hand_pose[frame_indices]
            object_pose = object_pose[frame_indices]  # [:, capture_filter, :]
        grasp_ycb_id = ycb_ids[meta["ycb_grasp_ind"]]
        object_mesh_files = [
            self._object_mesh_file(grasp_ycb_id)
        ]  # [self._object_mesh_file(ycb_id) for ycb_id in ycb_ids]

        # concatenation: note that the shape of object_pose and hand_pose is (x, 4, 7) and (x, 1, 51) respectively, so we need to reshape them to (x, 7) and (x, 51) respectively
        object_pose_1 = object_pose[:, meta["ycb_grasp_ind"], :].reshape(
            -1, 7
        )  # reshape the last two dimensions to one dimension
        hand_pose_1 = hand_pose.reshape(-1, 51)
        pose_pair = np.concatenate(
            (object_pose_1, hand_pose_1), axis=1
        )  # pose_paired_dataset[key]['pose'].shape = (x, 58)
        # import ipdb; ipdb.set_trace()

        ycb_data = dict(
            pose_pair=pose_pair,
            hand_pose=hand_pose_1,
            object_pose=object_pose_1,
            extrinsics=extrinsic_mat,
            ycb_ids=ycb_ids,
            ycb_grasp_ind=meta["ycb_grasp_ind"],
            hand_shape=mano_parameters,
            object_mesh_file=object_mesh_files[0],
            capture_name=capture_name,
            mano_side=mano_side,
            length=len(frame_indices),
        )
        return ycb_data

    def _filter_occluded_frame(self, frame_indices, mano_pose):
        """
        filter mano_pose = 0 frames
        """
        mano_not_occluded = (
            np.linalg.norm(mano_pose, axis=-1).reshape(-1) > 1e-3
        ).tolist()
        first_frame, last_frame, max_length = find_longest_true_sequence(
            mano_not_occluded
        )
        return frame_indices[first_frame:last_frame]

    def _filter_object_motion_frame(self, capture_filter, object_pose, frame_margin=40):
        frames = np.arange(0)
        for filter_id in capture_filter:
            filter_object_pose = object_pose[:, filter_id, :]
            object_move_list = []
            for frame in range(filter_object_pose.shape[0] - 2):
                object_move_list.append(
                    self.is_object_move(filter_object_pose[frame:, :])
                )
            if True not in object_move_list:
                continue
            first_frame = object_move_list.index(True)
            last_frame = len(object_move_list) - object_move_list[::-1].index(True) - 1
            start = max(0, first_frame - frame_margin)
            end = min(filter_object_pose.shape[0], last_frame + frame_margin)
            frames = np.arange(start, end)
            break
        return frames, filter_id

    @staticmethod
    def is_object_move(single_object_pose: np.ndarray):
        single_object_trans = single_object_pose[:, 4:]
        future_frame = min(single_object_trans.shape[0] - 1, 5)
        current_move = (
            np.linalg.norm(single_object_trans[1] - single_object_trans[0]) > 2e-2
        )
        future_move = (
            np.linalg.norm(single_object_trans[future_frame] - single_object_trans[0])
            > 5e-2
        )
        return current_move or future_move

    def _object_mesh_file(self, object_id):
        obj_file = (
            self._data_dir / "models" / YCB_CLASSES[object_id] / "textured_simple.obj"
        )
        return str(obj_file.resolve())

    def _load_camera_parameters(self):
        extrinsics = {}
        intrinsics = {}
        for cali_dir in self._calib_dir.iterdir():
            if not cali_dir.stem.startswith("extrinsics"):
                continue
            extrinsic_file = cali_dir / "extrinsics.yml"
            name = cali_dir.stem[len("extrinsics_") :]
            with extrinsic_file.open(mode="r") as f:
                extrinsic = yaml.load(f, Loader=yaml.FullLoader)
            extrinsics[name] = extrinsic

        intrinsic_dir = self._calib_dir / "intrinsics"
        for intrinsic_file in intrinsic_dir.iterdir():
            with intrinsic_file.open(mode="r") as f:
                intrinsic = yaml.load(f, Loader=yaml.FullLoader)
            name = intrinsic_file.stem.split("_")[0]
            x = intrinsic["color"]
            camera_mat = np.array(
                [[x["fx"], 0.0, x["ppx"]], [0.0, x["fy"], x["ppy"]], [0.0, 0.0, 1.0]]
            )
            intrinsics[name] = camera_mat

        return intrinsics, extrinsics

    def _load_mano(self):
        mano_parameters = {}
        for cali_dir in self._calib_dir.iterdir():
            if not cali_dir.stem.startswith("mano"):
                continue

            mano_file = cali_dir / "mano.yml"
            with mano_file.open(mode="r") as f:
                shape_parameters = yaml.load(f, Loader=yaml.FullLoader)
            mano_name = "_".join(cali_dir.stem.split("_")[1:])
            mano_parameters[mano_name] = np.array(shape_parameters["betas"])

        return mano_parameters


def loadit(dexycb_dir: str, hand_type: str):
    from collections import Counter

    dataset = DexYCBVideoDataset(dexycb_dir, hand_type)
    print(len(dataset))
    # import ipdb; ipdb.set_trace()

    ycb_names = []
    for i, data in enumerate(dataset):
        ycb_ids = data["ycb_ids"][0]
        ycb_names.append(YCB_CLASSES[ycb_ids])

    counter = Counter(ycb_names)
    print(counter)

    sample = dataset[0]
    print(sample.keys())
    return dataset
