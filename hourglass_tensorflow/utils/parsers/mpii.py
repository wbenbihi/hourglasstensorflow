"""This module provides helper functions to work with MPII raw structures"""

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Iterable
from typing import Optional

import scipy.io
from numpy import ndarray
from loguru import logger
from pydantic import BaseModel
from pydantic import ValidationError
from scipy.io.matlab._mio5_params import mat_struct as MatStruct

# region CONSTANTS


ADDITIONAL_ANNORECT_PARTS = [
    "head_r11",
    "head_r12",
    "head_r13",
    "head_r21",
    "head_r22",
    "head_r23",
    "head_r31",
    "head_r32",
    "head_r33",
    "part_occ1",
    "part_occ10",
    "part_occ2",
    "part_occ3",
    "part_occ4",
    "part_occ5",
    "part_occ6",
    "part_occ7",
    "part_occ8",
    "part_occ9",
    "torso_r11",
    "torso_r12",
    "torso_r13",
    "torso_r21",
    "torso_r22",
    "torso_r23",
    "torso_r31",
    "torso_r32",
    "torso_r33",
]

# endregion

# region Data Model - Annolist


class MPIIObjPos(BaseModel):
    x: int
    y: int


class MPIIAnnoPoint(BaseModel):
    id: Optional[int]
    x: Optional[int]
    y: Optional[int]
    is_visible: Optional[int]


class MPIIAnnorect(BaseModel):
    index: Optional[int]
    annopoints: Optional[List[MPIIAnnoPoint]]
    objpos: Optional[MPIIObjPos]
    scale: Optional[float]
    x1: Optional[int]
    y1: Optional[int]
    x2: Optional[int]
    y2: Optional[int]
    head_r11: Optional[float]
    head_r12: Optional[float]
    head_r13: Optional[float]
    head_r21: Optional[float]
    head_r22: Optional[float]
    head_r23: Optional[float]
    head_r31: Optional[float]
    head_r32: Optional[float]
    head_r33: Optional[float]
    part_occ1: Optional[float]
    part_occ10: Optional[float]
    part_occ2: Optional[float]
    part_occ3: Optional[float]
    part_occ4: Optional[float]
    part_occ5: Optional[float]
    part_occ6: Optional[float]
    part_occ7: Optional[float]
    part_occ8: Optional[float]
    part_occ9: Optional[float]
    torso_r11: Optional[float]
    torso_r12: Optional[float]
    torso_r13: Optional[float]
    torso_r21: Optional[float]
    torso_r22: Optional[float]
    torso_r23: Optional[float]
    torso_r31: Optional[float]
    torso_r32: Optional[float]
    torso_r33: Optional[float]


class MPIIAnnotation(BaseModel):
    index: int
    annorect: Optional[List[MPIIAnnorect]]
    frame_sec: Optional[int]
    image: str
    vididx: Optional[int]


# endregion

# region Data Model - Other


class MPIIAct(BaseModel):
    act_id: int
    act_name: Optional[List[str]]
    cat_name: Optional[List[str]]


class MPIIDataset(BaseModel):
    annolist: List[MPIIAnnotation]
    act: List[MPIIAct]
    img_train: List[int]
    single_person: List[List[int]]
    video_list: List[str]


class MPIIDatapoint(BaseModel):
    annolist: MPIIAnnotation
    act: MPIIAct
    img_train: int
    single_person: List[int]


# endregion

# region Custom Types

MatStructArray = Union[ndarray, Iterable[MatStruct]]

MPIIObject = Union[
    Dict[str, List],
    List[Dict[str, Union[Dict, int, List[int], str]]],
    MPIIDataset,
    List[MPIIDatapoint],
]

# endregion

# region Function - Helpers


def generic_condition(obj: MatStruct, key: str) -> bool:
    """Check if `scipy.io.matlab._mio5_params.mat_struct` object contains a key

    Two checks are performed:
        - Verify if the key is available in the object
        - If the key is available, checks for possible length of value

    Args:
        obj (MatStruct): The matlab structure object to verify
        key (str): the key to check

    Returns:
        bool: True is the conditions are met
    """
    return (key in obj.__dict__) and (0 not in obj.__dict__.get(key).shape)


def remove_null_keys(
    d: Dict, *, remove_null_keys: bool = True, strict_to_none: bool = True
) -> Dict:
    """Remove the keys with `None` value from a dictionary

    Args:
        d (Dict): dictionary
        remove_null_keys (bool, optional): True if you want to perform `None` value removal.
        Defaults to True.
        strict_to_none (bool, optional): True to only check for None and keep empty string, empty lists...

    Returns:
        Dict: dictionary with/without `None` value depending on `remove_null_keys`
    """
    return (
        {k: v for k, v in d.items() if (v is not None if strict_to_none else v)}
        if remove_null_keys
        else d
    )


# endregion

# region Function - Annolist


def parse_objpos(objpos: MatStruct, **kwargs) -> Dict:
    """Parse the `objpos` fields from a Matlab structure

    When the `objpos` is set it contains 2 fields:
        - `x` as integer
        - `y` as integer

    Args:
        objpos (MatStruct): matlab structure

    Returns:
        Dict: parsed matlab structure
    """
    return remove_null_keys(
        {
            "x": int(objpos.x[0][0]),
            "y": int(objpos.y[0][0]),
        },
        **kwargs,
    )


def parse_is_visible(is_visible: ndarray) -> int:
    """Parse the `is_visible` fields from an Annopoint Matlab structure

    Args:
        is_visible (ndarray): ndarray of shape (1, 1) or (1, )

    Returns:
        int: The `is_visible` value as 0/1
    """
    if len(is_visible.shape) == 2:
        return is_visible[0][0]
    else:
        return is_visible[0]


def parse_annopoints(points: Iterable[MatStruct], **kwargs) -> List[Dict]:
    """Parse the `annopoints` fields from a Matlab structure

    When the `annopoints` is set it contains 4 fields:
        - `id` as integer, joint id
        - `x` as integer, X coordinate
        - `y` as integer, Y coordinate
        - `is_visible` as integer, boolean as integer. Optional

    Args:
        points (Iterable[MatStruct]): matlab structure

    Returns:
        List[Dict]: parsed matlab structure as dictionary
    """
    return [
        remove_null_keys(
            {
                "id": (int(point.id[0][0]) if generic_condition(point, "id") else None),
                "x": (int(point.x[0][0]) if generic_condition(point, "x") else None),
                "y": (int(point.y[0][0]) if generic_condition(point, "y") else None),
                "is_visible": (
                    int(parse_is_visible(point.is_visible))
                    if generic_condition(point, "is_visible")
                    else None
                ),
            },
            **kwargs,
        )
        for point in points
    ]


def parse_additional_annorect_item(person: MatStruct, **kwargs) -> Dict:
    """Parse the additional `annorect` fields from a Matlab structure

    This function looks at misc keys from the `annorect` structure:
        Check Notes for further details

    Args:
        person (MatStruct): matlab structure

    Returns:
        Dict: parsed matlab structure as dictionary

    Notes:
        Additional data from the `annorect` structure might be:
        - head_r11 (float):
        - head_r12 (float):
        - head_r13 (float):
        - head_r21 (float):
        - head_r22 (float):
        - head_r23 (float):
        - head_r31 (float):
        - head_r32 (float):
        - head_r33 (float):
        - part_occ1 (float):
        - part_occ10 (float):
        - part_occ2 (float):
        - part_occ3 (float):
        - part_occ4 (float):
        - part_occ5 (float):
        - part_occ6 (float):
        - part_occ7 (float):
        - part_occ8 (float):
        - part_occ9 (float):
        - torso_r11 (float):
        - torso_r12 (float):
        - torso_r13 (float):
        - torso_r21 (float):
        - torso_r22 (float):
        - torso_r23 (float):
        - torso_r31 (float):
        - torso_r32 (float):
        - torso_r33 (float):
    """
    return remove_null_keys(
        {
            part: float(person.__dict__.get(part)[0])
            if generic_condition(person, part)
            else None
            for part in ADDITIONAL_ANNORECT_PARTS
        },
        **kwargs,
    )


def parse_annorect_item(index: int, person: MatStruct, **kwargs) -> Dict:
    """Parse the `annorect` fields from a Matlab structure

    When the `annopoints` is set it contains 8 main fields:
        - `index` as integer, person id in the picture
        - `annopoints` as list, list of joint coordinates
        - `x1` as integer, TopLeft BBox corner X coordinate
        - `y1` as integer, TopLeft BBox corner Y coordinate
        - `x2` as integer, BottomRight BBox corner X coordinate
        - `y2` as integer, BottomRight BBox corner Y coordinate
        - `objpos` as dict,
        - `scale` as float,

    Args:
        index (int): person index with the available persons on the image
        person (MatStruct): matlab structure

    Returns:
        Dict: parsed matlab structure as dictionary
    """
    return remove_null_keys(
        {
            **{
                "index": index,
                "annopoints": (
                    parse_annopoints(person.annopoints[0][0].point[0])
                    if generic_condition(person, "annopoints")
                    else None
                ),
                "objpos": (
                    parse_objpos(person.objpos[0][0])
                    if generic_condition(person, "objpos")
                    else None
                ),
                "scale": (
                    float(person.scale[0][0])
                    if generic_condition(person, "scale")
                    else None
                ),
                "x1": (
                    int(person.x1[0][0]) if generic_condition(person, "x1") else None
                ),
                "y1": (
                    int(person.x1[0][0]) if generic_condition(person, "y1") else None
                ),
                "x2": (
                    int(person.x2[0][0]) if generic_condition(person, "x2") else None
                ),
                "y2": (
                    int(person.x2[0][0]) if generic_condition(person, "y2") else None
                ),
            },
            **parse_additional_annorect_item(person),
        },
        **kwargs,
    )


def parse_annorect(annorect: MatStructArray, **kwargs) -> List[Dict]:
    """Parse a list of `annorect` items

    Args:
        annorect (Union[ndarray, Iterable[MatStruct]] as MatStructArray): list of `annorect` matlab structure

    Returns:
        List[Dict]: parsed matlab structure as List of dictionaries
    """
    return [
        parse_annorect_item(i, person, **kwargs) for i, person in enumerate(annorect)
    ]


def parse_annolist(annolist: MatStructArray, **kwargs) -> List[Dict]:
    """Parse a list of `annolist` items

    When the `annolist` is set it contains 5 main fields:
        - `index` as integer, image index
        - `annorect` as list, list of body annotations for persons in image
        - `image` as str, image name
        - `vididx` as integer, video index. Optional
        - `frame_sec` as integer, image position in video in seconds. Optional

    Args:
        annolist (Union[ndarray, Iterable[MatStruct]] as MatStructArray): list of `annolist` matlab structure

    Returns:
        List[Dict]: parsed matlab structure as List of dictionaries
    """
    return [
        remove_null_keys(
            {
                "index": int(i),
                "frame_sec": (
                    int(item.frame_sec[0][0]) if 0 not in item.frame_sec.shape else None
                ),
                "image": str(item.image[0][0].name[0]),
                "vididx": (
                    int(item.vididx[0][0]) if 0 not in item.vididx.shape else None
                ),
                "annorect": (
                    parse_annorect(
                        item.annorect[0] if 0 not in item.annorect.shape else []
                    )
                ),
            },
            **kwargs,
        )
        for i, item in enumerate(annolist)
    ]


# endregion

# region Function - Img Train


def parse_img_train(img_train: ndarray, **kwargs) -> List[int]:
    """Parse a list of integer from a matlab structure

    The `img_train` stores a 2D array with a single dimension of interest.
    It contains a list of 0/1 as boolean value for test/train

    Args:
        img_train (ndarray): _description_

    Returns:
        List[int]: _description_
    """
    return img_train[0].astype(int).tolist()


# endregion

# region Function - Single Person


def parse_single_person(single_person: ndarray, **kwargs) -> List[List[int]]:
    """Parse a list of integer from a matlab structure

    The `single_person` is a 2D array with a single dimension of interest.
    It contains a list of rectangle identifier of sufficiently separated person

    Args:
        single_person (ndarray): _description_

    Returns:
        List[List[int]]: _description_
    """
    return [
        [int(elm[0]) for elm in person[0]] if 0 not in person[0].shape else []
        for person in single_person
    ]


# endregion

# region Function - Video List


def parse_video_list(
    video_list: Union[ndarray, Iterable[Iterable[str]]], **kwargs
) -> List[str]:
    """Parse a list of string from a matlab structure

    The `video_list` is a 2D array with a single dimension of interest.
    It contains a list of Youtube video identifier

    Args:
        video_list (Union[ndarray, Iterable[Iterable[str]]]): _description_

    Returns:
        List[str]: _description_
    """
    return [str(video_name[0]) for video_name in video_list.tolist()]


# endregion

# region Function - Act


def parse_act(act: MatStructArray, **kwargs) -> List[Dict]:
    """Parse a list of string from a matlab structure

    The `act` is a 2D array with a single dimension of interest.
    It contains a list of action describer keywords & categories

    Args:
        act (Union[ndarray, Iterable[MatStruct]] as MatStructArray): _description_

    Returns:
        List[Dict]: _description_
    """
    return [
        remove_null_keys(
            {
                "act_id": int(a.act_id[0][0]),
                "act_name": (
                    a.act_name.tolist()[0].split(", ") if len(a.act_name) else []
                ),
                "cat_name": (
                    a.cat_name.tolist()[0].split(", ") if len(a.cat_name) else []
                ),
            },
            **kwargs,
        )
        for a in act
    ]


# endregion

# region Main Function


def open_mat_file(filename: str) -> Dict:
    """Open a Matlab `.mat` file

    Args:
        filename (str): Matlab file name

    Returns:
        Dict: Matlab object as dictionary
    """
    return scipy.io.loadmat(filename, struct_as_record=False)


def parse_mpii(
    mpii_annot_file: str,
    test_parsing: bool = False,
    verify_len: bool = True,
    return_as_struct: bool = False,
    zip_struct: bool = False,
    verbose: bool = False,
    **kwargs,
) -> MPIIObject:
    """Parse a MPII Matlab Structure

    Args:
        mpii_annot_file (str): MPII `.mat` file name
        test_parsing (bool, optional): Test if the parsing to pydantic is possible.
            Redundant is set to True with `return_as_struct`. Defaults to False.
        verify_len (bool, optional): Check if structures' length match. Defaults to True.
        return_as_struct (bool, optional): Whether or not to return result as a pydantic object.
            Defaults to False.
        zip_struct (bool, optional): Return the objet as `List[Dict]` instead of `Dict[str, List]`.
            Combined with `return_as_struct`,
            returns the object as `List[MPIIDatapoint]` instead of `MPIIDataset`.
            Defaults to False.

    Raises:
        IndexError: When `verify_len = True` and list does not match in length.
        ValidationError: When `test_parsing = True` and object is not parsable to `MPIIDataset`.

    Returns:
        Union[
            Dict[str, List],
            List[Dict[str, Union[Dict, int, List[int], str]]],
            MPIIDataset, List[MPIIDatapoint]
        ](MPIIObject): Parsed MPII Matlab structure
    """
    mat = open_mat_file(mpii_annot_file)
    # Get the relevant data
    mat_struct = mat["RELEASE"][0][0]
    # Destructure object
    annolist: ndarray = mat_struct.annolist[0]
    act: ndarray = mat_struct.act[:, 0]
    img_train: ndarray = mat_struct.img_train
    single_person: ndarray = mat_struct.single_person
    video_list: ndarray = mat_struct.video_list[0]
    # Parse data
    parsed_annolist = parse_annolist(annolist, **kwargs)
    parsed_img_train = parse_img_train(img_train, **kwargs)
    parsed_single_person = parse_single_person(single_person, **kwargs)
    parsed_video_list = parse_video_list(video_list, **kwargs)
    parsed_act = parse_act(act, **kwargs)
    # Construct Object
    mpii_dict = dict(
        annolist=parsed_annolist,
        img_train=parsed_img_train,
        single_person=parsed_single_person,
        video_list=parsed_video_list,
        act=parsed_act,
    )
    return_obj = mpii_dict
    # Test List Size
    if verbose:
        logger.debug(f"len(parsed_annolist) = {len(parsed_annolist)}")
        logger.debug(f"len(parsed_img_train) = {len(parsed_img_train)}")
        logger.debug(f"len(parsed_single_person) = {len(parsed_single_person)}")
        logger.debug(f"len(parsed_video_list) = {len(parsed_video_list)}")
        logger.debug(f"len(parsed_act) = {len(parsed_act)}")
    if verify_len and not (
        len(parsed_annolist)
        == len(parsed_img_train)
        == len(parsed_single_person)
        == len(parsed_act)
    ):
        # We don't check for video_list length since it does not match the others
        raise IndexError("The parsed list does not have matching size")
    # Test if object is parsable as MPIIDataset(BaseModel)
    if test_parsing:
        try:
            MPIIDataset.parse_obj(mpii_dict)
        except ValidationError as e:
            raise e
    # Construct return object
    if zip_struct:
        # In this specific case we don't store video_list anymore
        return_obj = [
            dict(
                annolist=_annolist,
                img_train=_img_train,
                single_person=_single_person,
                act=_act,
            )
            for (_annolist, _img_train, _single_person, _act) in zip(
                parsed_annolist,
                parsed_img_train,
                parsed_single_person,
                parsed_act,
            )
        ]
        if return_as_struct:
            return_obj = [
                MPIIDatapoint.parse_obj(datapoint) for datapoint in return_obj
            ]
    else:
        if return_as_struct:
            return_obj = MPIIDataset.parse_obj(mpii_dict)
    return return_obj


# endregion
