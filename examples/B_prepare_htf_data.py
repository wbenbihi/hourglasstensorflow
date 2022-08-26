from typing import Dict
from typing import List

import numpy as np
from loguru import logger
from pydantic import parse_file_as

from hourglass_tensorflow.types import HTFPersonDatapoint

HTF_JSON = "data/htf.ignore.json"

if __name__ == "__main__":
    # Parse file as list of records
    logger.info(f"Reading HTF data at {HTF_JSON}")
    data = parse_file_as(List[HTFPersonDatapoint], HTF_JSON)
    # Compute Stats
    ## Average number of joints and average number of visible joints
    num_joints = [len(d.joints) for d in data]
    num_visible_joints = [len([j for j in d.joints if j.visible]) for d in data]
    avg_joints_per_sample = np.mean(num_joints)
    avg_visible_joints_per_sample = np.mean(num_visible_joints)
    ## Joint ID distribution
    joints_id = [(j.id, j.visible) for d in data for j in d.joints]
    only_visible_joints_id = [jid for jid, j_visible in data if j_visible]
    # Prepare data as table

    # Write Transformed data
