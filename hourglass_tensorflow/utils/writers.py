import json
import pickle
from typing import Dict
from typing import List
from typing import Union

import yaml
from pydantic import BaseModel

WRITER_MAPPER = {
    "yaml": lambda data, fp: yaml.dump(data, fp),
    "yml": lambda data, fp: yaml.dump(data, fp),
    "json": lambda data, fp: json.dump(data, fp),
    "pickle": lambda data, fp: pickle.dump(data, fp),
}


def common_write(
    obj: Union[List, BaseModel, Dict], path: str, force_dict_struct: bool = False
):
    """Write a given object to file path

    Args:
        obj (Union[List, BaseModel, Dict]): Object to write
        path (str): Path to write the object
        force_dict_struct (bool, optional): In case you pickle a BaseModel,
             you can force the object to be stored as built-in types
            Defaults to False.

    Raises:
        KeyError: _description_
    """
    try:
        # Check if the file extension is supported
        file_extension = path.split(".")[-1]
        extension_mapper = WRITER_MAPPER[file_extension]
    except KeyError:
        raise KeyError(
            "Output file should be of following types [.yml, .yaml, .json, .pickle]"
        )

    output_obj = obj
    if force_dict_struct or file_extension in ["yml", "yaml", "json"]:
        # We will cast any pydantic.BaseModel if
        #   - The force_dict_struct is set to True (pickle only)
        #   - The output file type require to write a dict
        if isinstance(obj, BaseModel):
            output_obj = obj.dict()
        elif isinstance(obj, list):
            output_obj = [o.dict() if isinstance(o, BaseModel) else o for o in obj]
    # We write the file
    with open(path, "w") as f:
        extension_mapper(output_obj, f)
