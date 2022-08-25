import os
import json

import pytest
from loguru import logger
from pydantic import BaseModel

from hourglass_tensorflow.utils import common_write

WRITE_TO_FILE = "write.ignore.{EXT}"


@pytest.fixture(scope="function")
def model():
    class TestModel(BaseModel):
        value: int

    return TestModel


def test_common_write_unsupported_config_file_type(model):
    # Init
    ext = "txt"
    filename = WRITE_TO_FILE.format(EXT=ext)
    obj = model(value=3)
    with pytest.raises(KeyError):
        common_write(obj=obj, path=filename)
    # Remove file when Over
    if os.path.exists(filename):
        os.remove(filename)


@pytest.mark.parametrize("ext", ["yaml", "yml", "json"])
def test_common_write_supported_config_file_type(model, ext):
    # Init
    filename = WRITE_TO_FILE.format(EXT=ext)
    obj = model(value=3)
    common_write(obj=obj, path=filename)
    # Remove file when Over
    assert os.path.exists(filename), "File was not written"
    if os.path.exists(filename):
        os.remove(filename)


def test_common_write(model):
    # Init
    filename = WRITE_TO_FILE.format(EXT="json")

    # Test Basic Object
    obj = model(value=3)
    common_write(obj=obj, path=filename)
    with open(filename, "r") as f:
        data = json.load(f)
    assert (
        "value" in data and data["value"] == 3
    ), "The object was not properly serialized"

    # Test list of objects
    obj = [model(value=3), model(value=4), model(value=4)]
    common_write(obj=obj, path=filename)
    with open(filename, "r") as f:
        data = json.load(f)
    assert [
        "value" in data[i] and data[i]["value"] == o.value for i, o in enumerate(obj)
    ], "The object was not properly serialized"

    # Test list of dictionaries
    obj = [dict(value=3), dict(value=4), dict(value=4)]
    common_write(obj=obj, path=filename)
    with open(filename, "r") as f:
        data = json.load(f)
    assert [
        "value" in data[i] and data[i]["value"] == o["value"] for i, o in enumerate(obj)
    ], "The object was not properly serialized"

    # Remove file when Over
    assert os.path.exists(filename), "File was not written"
    if os.path.exists(filename):
        os.remove(filename)
