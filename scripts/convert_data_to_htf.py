import json
from typing import Dict
from typing import List

from loguru import logger

from hourglass_tensorflow.utils.parsers import MPIIDatapoint
from hourglass_tensorflow.utils.parsers import parse_mpii
from hourglass_tensorflow.utils.writers import common_write
from hourglass_tensorflow.utils.parsers.htf import from_train_mpii_to_htf_data

MAT_FILE = "data/mpii.ignore.mat"
HTF_JSON = "data/HTF_DATA.ignore.json"

if __name__ == "__main__":
    # Parse file as list of records
    logger.info(f"Parse MPII data at {MAT_FILE}")
    data: List[Dict] = parse_mpii(
        mpii_annot_file=MAT_FILE,
        test_parsing=False,
        verify_len=False,
        return_as_struct=False,
        zip_struct=True,
        remove_null_keys=False,
    )
    # Cast records as List[MPIIDatapoint]
    logger.info("Cast data as record of structures")
    datapoints = [MPIIDatapoint.parse_obj(datapoint) for datapoint in data]
    logger.info("Convert from MPII to HTF")
    htf_data, _ = from_train_mpii_to_htf_data(data=datapoints, require_stats=True)
    # Write Transform data
    logger.info(f"Write HTF data to {HTF_JSON}")
    common_write(htf_data, HTF_JSON)
