from typing import List
from typing import Tuple
from typing import Union

from hourglass_tensorflow.types import HTFPoint
from hourglass_tensorflow.types import HTFPersonBBox
from hourglass_tensorflow.types import HTFPersonJoint
from hourglass_tensorflow.types import HTFPersonDatapoint
from hourglass_tensorflow.utils.parsers import mpii


def _check_mpii_format(data: mpii.MPIIObject) -> Tuple[bool, bool]:
    """Check if the MPII compliant object is structured and has record format

    Args:
        data (mpii.MPIIObject): _description_

    Raises:
        TypeError: data is compliant with no mpii.MPIIObject structure

    Returns:
        Tuple[bool, bool]: return is_struct, is_record
    """
    try:
        if isinstance(data, list):
            is_record = True
            if isinstance(data[0], mpii.MPIIDatapoint):
                is_struct = True
            else:
                is_struct = False
        elif isinstance(data, mpii.MPIIDataset):
            is_record = False
            is_struct = True
        elif isinstance(data, dict):
            is_struct = False
            is_record = False
            if not (
                "annolist" in data
                and "act" in data
                and "img_train" in data
                and "single_person" in data
                and "video_list" in data
            ):
                raise TypeError
        else:
            raise TypeError
    except TypeError:
        raise TypeError(
            "The input data type was not recognised. Ensure it is compliant with mpii.MPIIObject"
        )
    return is_struct, is_record


def _convert_mpii_to_struct_record(
    data: mpii.MPIIObject, is_struct: bool, is_record: bool
) -> List[mpii.MPIIDatapoint]:
    _converted = data
    if not is_struct:
        # Convert to Struct
        if not is_record:
            # Convert to Record
            _converted = [
                dict(
                    annolist=_annolist,
                    img_train=_img_train,
                    single_person=_single_person,
                    act=_act,
                )
                for (_annolist, _img_train, _single_person, _act) in zip(
                    _converted.get("annolist"),
                    _converted.get("img_train"),
                    _converted.get("single_person"),
                    _converted.get("act"),
                )
            ]
        _converted = [mpii.MPIIDatapoint.parse_obj(d) for d in _converted]
    else:
        # Is Struct
        if not is_record:
            # Convert to Struct
            _converted = [
                mpii.MPIIDatapoint.parse_obj(
                    dict(
                        annolist=_annolist,
                        img_train=_img_train,
                        single_person=_single_person,
                        act=_act,
                    )
                )
                for (_annolist, _img_train, _single_person, _act) in zip(
                    _converted.annolist,
                    _converted.img_train,
                    _converted.single_person,
                    _converted.act,
                )
            ]
    return _converted


def from_train_mpii_to_htf_data(
    data: mpii.MPIIObject, require_stats=False
) -> Union[List[HTFPersonDatapoint], Tuple[List[HTFPersonDatapoint], Tuple]]:
    # TODO: Works only if `remove_null_keys` was set to False
    # while parsing MPII data.
    # First check if the data is record wise
    is_struct, is_record = _check_mpii_format(data)
    # Format to structured records
    records = _convert_mpii_to_struct_record(
        data, is_struct=is_struct, is_record=is_record
    )
    # Filter only train datapoints since test does not contain data
    train = [r for r in records if r.img_train == 1]
    # Filter train samples with at least 1 joint annotation
    train_with_annopoints = [
        p for p in train if any([r for r in p.annolist.annorect if r.annopoints])
    ]
    # Cast data to List[HTFPersonDatapoint]
    records_to_return = [
        HTFPersonDatapoint(
            is_train=sample.img_train,
            image_id=sid,
            source_image=sample.annolist.image,
            person_id=pid,
            bbox=HTFPersonBBox(
                top_left=HTFPoint(x=person.x1, y=person.y1),
                bottom_right=HTFPoint(x=person.x2, y=person.y2),
            ),
            joints=[
                HTFPersonJoint(
                    x=joint.x, y=joint.y, id=joint.id, visible=bool(joint.is_visible)
                )
                for joint in person.annopoints
                if isinstance(joint, mpii.MPIIAnnoPoint)
            ],
            center=HTFPoint.parse_obj(person.objpos) if person.objpos else None,
            scale=person.scale,
        )
        for sid, sample in enumerate(train_with_annopoints)
        for pid, person in enumerate(sample.annolist.annorect)
        if isinstance(person, mpii.MPIIAnnorect) and person.annopoints is not None
    ]
    if require_stats:
        stats = {
            "a_source_data": len(data),
            "b_train_data": len(train),
            "c_train_with_annopoints": len(train_with_annopoints),
            "d_final_records": len(records_to_return),
        }
        return records_to_return, stats
    else:
        return records_to_return
