import os
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Literal

import matplotlib.pyplot as plt

from hourglass_tensorflow.types import HTFPersonDatapoint

# region Color Maps
COLOR_MAP = {
    0: "red",
    1: "blue",
    2: "green",
    3: "yellow",
    4: "purple",
    5: "cyan",
    6: "magenta",
    7: "orange",
    8: "dodgerblue",
    9: "lime",
    10: "gold",
    11: "violet",
    12: "pink",
    13: "teal",
    14: "lightcoral",
    15: "peru",
}

LIMB_COLORS = {
    "head": COLOR_MAP[0],
    "r_ankle_knee": COLOR_MAP[1],
    "r_knee_hip": COLOR_MAP[2],
    "r_hip_pelvis": COLOR_MAP[3],
    "l_hip_pelvis": COLOR_MAP[4],
    "l_knee_hip": COLOR_MAP[5],
    "l_ankle_knee": COLOR_MAP[6],
    "pelvis_thorax": COLOR_MAP[7],
    "thorax_neck": COLOR_MAP[8],
    "r_wrist_elbow": COLOR_MAP[9],
    "r_elbow_shoulder": COLOR_MAP[10],
    "l_elbow_shoulder": COLOR_MAP[11],
    "l_wrist_elbow": COLOR_MAP[12],
    "l_shoulder_neck": COLOR_MAP[13],
    "r_shoulder_neck": COLOR_MAP[14],
}

# endregion

# region Limbs
LIMBS = {
    "head": (8, 9),
    "r_ankle_knee": (0, 1),
    "r_knee_hip": (1, 2),
    "r_hip_pelvis": (2, 6),
    "l_hip_pelvis": (3, 6),
    "l_knee_hip": (4, 3),
    "l_ankle_knee": (5, 4),
    "pelvis_thorax": (6, 7),
    "thorax_neck": (7, 8),
    "r_wrist_elbow": (10, 11),
    "r_elbow_shoulder": (11, 12),
    "l_elbow_shoulder": (13, 14),
    "l_wrist_elbow": (14, 15),
    "l_shoulder_neck": (13, 8),
    "r_shoulder_neck": (12, 8),
}

# endregion

# region Type Hints
JointPlotModes = Union[Literal["joints"], Literal["limbs"], Literal["dots"]]

# endregion


def plot_sample_with_joint(
    image,
    joints: Dict[int, Tuple[int, int]],
    colors=COLOR_MAP,
    mode: JointPlotModes = "joints",
    limbs: Dict[str, Tuple[int, int]] = LIMBS,
    limb_colors: Dict[str, str] = LIMB_COLORS,
):
    scatters = []
    lines = []
    if "dots" in mode:
        scatters = [([[j[0]], [j[1]]], {"color": colors[0]}) for j in joints.values()]
    if "joints" in mode:
        scatters += [
            ([[j[0]], [j[1]]], {"color": colors[jid]}) for jid, j in joints.items()
        ]
    if "limbs" in mode:
        lines += [
            (
                [
                    [joints[limb_joints[0]][0], joints[limb_joints[1]][0]],
                    [joints[limb_joints[0]][1], joints[limb_joints[1]][1]],
                ],
                {"color": limb_colors[limb_name]},
            )
            for limb_name, limb_joints in limbs.items()
            if joints.get(limb_joints[0]) and joints.get(limb_joints[1])
        ]

    fig, ax = plt.subplots()
    _ = ax.imshow(image)
    for pos_scatter, kw_scatter in scatters:
        ax.scatter(*pos_scatter, **kw_scatter)
    for pos_line, kw_lines in lines:
        ax.plot(*pos_line, **kw_lines)
    plt.show()


def plot_datapoint(
    datapoint: HTFPersonDatapoint,
    colors=COLOR_MAP,
    image_path="",
    mode: JointPlotModes = "joints",
    limbs=LIMBS,
    limb_colors=LIMB_COLORS,
):
    image_path = os.path.join(image_path, datapoint.source_image)
    image = plt.imread(image_path)
    if isinstance(datapoint.joints, list):
        joints = {j.id: (j.x, j.y) for j in datapoint.joints}
    else:
        joints = datapoint.joints
    plot_sample_with_joint(
        image=image,
        joints=joints,
        colors=colors,
        mode=mode,
        limbs=limbs,
        limb_colors=limb_colors,
    )
