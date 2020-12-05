import scipy.io
from utils.mpii_mat_handler import parse_annolist, parse_act, parse_video_list, parse_single_person, parse_img_train
import re
from utils.decorators import conditional_decorator
import pandas as pd


class MPIIDatasetParser:
    def __init__(self, mat_path):
        self.mat_path = mat_path
        self.data_loaded = False
        self.data_parsed = False
        self.data_formed = False

        self._read_mat()

    def __call__(self, path):
        return self.parse_data().formalize_data().export_data(path)

    def _read_mat(self):
        self.mat = scipy.io.loadmat(self.mat_path, struct_as_record=False)
        self.release_mat = self.mat["RELEASE"][0][0]
        self.data_loaded = True

    @property
    def is_loaded(self):
        return self.data_loaded

    @property
    def is_parsed(self):
        return self.data_parsed

    @property
    def is_formed(self):
        return self.data_formed

    @property
    def _img_train(self):
        return self.release_mat.__dict__.get("img_train")[0]

    @property
    def _video_list(self):
        return self.release_mat.__dict__.get("video_list")[0]

    @property
    def _mpii_version(self):
        return self.release_mat.__dict__.get("version")[0]

    @property
    def _annolist(self):
        return self.release_mat.__dict__.get("annolist")[0]

    @property
    def _single_person(self):
        return self.release_mat.__dict__.get("single_person")

    @property
    def _act(self):
        return self.release_mat.__dict__.get("act")

    def _parse_video_list(self):
        return parse_video_list(self._video_list)

    def _parse_img_train(self):
        return parse_img_train(self._img_train)

    def _parse_single_person(self):
        return parse_single_person(self._single_person)

    def _parse_annolist(self):
        return parse_annolist(self._annolist)

    def _parse_act(self):
        return parse_act(self._act)

    def parse_data(self):
        self.mpii_data = {}
        self.mpii_data["annolist"] = self._parse_annolist()
        self.mpii_data["act"] = self._parse_act()
        self.mpii_data["img_train"] = self._parse_img_train()
        self.mpii_data["single_person"] = self._parse_single_person()
        self.mpii_data["video_list"] = self._parse_video_list()
        self.data_parsed = True
        return self

    def _formalize_img_df(self):
        img_df = pd.DataFrame.from_dict(
            [ann["annolist"] for ann in self.mpii_data["annolist"]]
        )  # .set_index('imgidx')
        img_df["is_train"] = self.mpii_data["img_train"]
        img_df["frame_sec"] = img_df["frame_sec"].fillna(-1).astype(int)
        img_df["vididx"] = img_df["vididx"].fillna(-1).astype(int)
        _ = img_df.apply(
            lambda x: [
                person["person"].update({"imgidx": x.imgidx}) for person in x.annorect
            ]
            if x.annorect is not None
            else 0,
            axis=1,
        )
        return img_df

    def _formalize_single_person_df(self, img_df):
        persons = img_df.query("annorect.notna()").annorect.sum()
        person_df = pd.DataFrame([person["person"] for person in persons])
        person_df = person_df.assign(
            objpos_x=person_df.objpos.map(lambda x: x["x"]),
            objpos_y=person_df.objpos.map(lambda x: x["y"]),
        )

        joint_dict = {}
        for idx_joint in range(16):
            joint_dict.update(
                {
                    f"joint_{idx_joint}": person_df.points.apply(
                        lambda x: [
                            joint["point"]
                            for joint in x
                            if joint["point"]["id"] == idx_joint
                        ]
                        if x is not None
                        else None
                    ).str[0]
                }
            )
        person_df = person_df.assign(**joint_dict)

        for joint in range(16):
            joint_dict = {}
            series = person_df[f"joint_{joint}"]
            joint_dict = {
                f"joint_{joint}_x": series.str["x"],
                f"joint_{joint}_y": series.str["y"],
                f"joint_{joint}_id": series.str["id"],
                f"joint_{joint}_is_visible": series.str["is_visible"],
            }
            person_df = person_df.assign(**joint_dict)

        HEAD_COLUMNS = [
            "imgidx",
            "ridx",
            "x1",
            "x2",
            "y1",
            "y2",
            "scale",
            "objpos_x",
            "objpos_y",
        ]
        JOINT_COLUMNS = [
            col
            for col in person_df.columns
            if re.match("joint_[0-9]*_(x|y|id|is_visible)", col)
        ]
        single_person_df = person_df[HEAD_COLUMNS + JOINT_COLUMNS]

        return single_person_df

    def _formalize_act_df(self):
        act_df = pd.DataFrame([act["act"] for act in self.mpii_data["act"]])
        act_df = act_df.assign(act_name=act_df.act_name.str.join(", "))
        return act_df

    def _formalize_video_df(self):
        video_df = pd.DataFrame(
            [vl["video"] for vl in self.mpii_data["video_list"]]
        ).rename(columns={"videoidx": "vididx"})
        video_df.vididx += 1
        return video_df

    def _formalize_mpii_df(self, img_df, single_person_df, video_df, act_df):
        mpii_dataset = (
            img_df[["imgidx", "image", "frame_sec", "vididx", "is_train"]]
            .merge(single_person_df, on="imgidx", how="inner")
            .merge(video_df, on="vididx", how="left")
            .merge(act_df, on="imgidx", how="left")
        )
        return mpii_dataset

    def formalize_data(self):
        img_df = self._formalize_img_df()
        single_person_df = self._formalize_single_person_df(img_df)
        act_df = self._formalize_act_df()
        video_df = self._formalize_video_df()
        self.mpii_dataset = self._formalize_mpii_df(
            img_df, single_person_df, video_df, act_df
        )
        return self
    
    def export_data(self, path):
        self.mpii_dataset.to_csv(path, sep=';', index=False)
        return self


if __name__ == "__main__":
    from config import CFG
    import os

    ROOT_FOLDER = CFG.ROOT_FOLDER
    DATA_FOLDER = "data"
    MPII_MAT = "mpii_human_pose_v1_u12_1.mat"
    MAT_PATH = os.path.join(ROOT_FOLDER, DATA_FOLDER, MPII_MAT)

    mpii_dataset_parser = MPIIDatasetParser(mat_path=MAT_PATH)

