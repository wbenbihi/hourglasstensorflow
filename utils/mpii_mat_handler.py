import json
import numpy as np


def generic_condition(obj, key):
    return (key in obj.__dict__) and (0 not in obj.__dict__.get(key).shape)


def parse_point(point):
    return {
        "point": {
            "x": float(point.__dict__.get("x")[0][0])
            if generic_condition(point, "x")
            else None,
            "y": float(point.__dict__.get("y")[0][0])
            if generic_condition(point, "y")
            else None,
            "id": int(point.__dict__.get("id")[0][0])
            if generic_condition(point, "id")
            else None,
            "is_visible": int(point.__dict__.get("is_visible")[0][0])
            if generic_condition(point, "is_visible")
            else None,
        }
    }


def parse_person(person, idx):
    return {
        "person": {
            "ridx": int(idx),
            "x1": int(person.__dict__.get("x1")[0][0])
            if generic_condition(person, "x1")
            else None,
            "x2": int(person.__dict__.get("x2")[0][0])
            if generic_condition(person, "x2")
            else None,
            "y1": int(person.__dict__.get("y1")[0][0])
            if generic_condition(person, "y1")
            else None,
            "y2": int(person.__dict__.get("y2")[0][0])
            if generic_condition(person, "y2")
            else None,
            "scale": float(person.__dict__.get("scale")[0][0])
            if generic_condition(person, "scale")
            else None,
            "objpos": {
                "x": int(person.__dict__.get("objpos")[0][0].__dict__.get("x")[0][0])
                if generic_condition(person, "objpos")
                else None,
                "y": int(person.__dict__.get("objpos")[0][0].__dict__.get("y")[0][0])
                if generic_condition(person, "objpos")
                else None,
            },
            "points": [
                parse_point(point)
                for point in person.__dict__.get("annopoints")[0][0].__dict__["point"][
                    0
                ]
            ]
            if generic_condition(person, "annopoints")
            else None,
        }
    }


def parse_persons(persons):
    return [parse_person(person, i) for i, person in enumerate(persons)]


def parse_annolist(annolist):
    return [
        {
            'annolist':{
                'imgidx':int(i),
                'image':item.__dict__.get('image')[0][0].__dict__.get('name')[0],
                'annorect':parse_persons(item.__dict__.get('annorect')[0]) if 0 not in item.__dict__.get('annorect').shape else None,
                'frame_sec':int(item.__dict__.get('frame_sec')[0][0]) if 0 not in item.__dict__.get('frame_sec').shape else None,
                'vididx':int(item.__dict__.get('vididx')[0][0]) if 0 not in item.__dict__.get('vididx').shape else None,
            }
        }
        for i, item in enumerate(annolist)
    ]


def parse_act(act):
    return [
        {
            "act": {
                "imgidx": int(i),
                "cat_name": elem[0].__dict__.get("cat_name")[0]
                if len(elem[0].__dict__.get("cat_name"))
                else None,
                "act_name": elem[0].__dict__.get("act_name")[0].split(", ")
                if len(elem[0].__dict__.get("act_name"))
                else None,
                "act_id": int(elem[0].__dict__.get("act_id")[0][0]),
            }
        }
        for i, elem in enumerate(act)
    ]


def parse_single_person(single_person):
    return [
        {
            'single_person':{
                'imgidx':int(i),
                'ridx': [int(elm[0]) for elm in item[0]] if 0 not in item[0].shape else None
            }
        }
        for i, item in enumerate(single_person)
    ]


def parse_video_list(video_list):
    return [{'video': {'videoidx':int(i), 'video_list':item[0]}} for i, item in enumerate(video_list)]


def parse_img_train(img_train):
    return img_train.astype(int).tolist()