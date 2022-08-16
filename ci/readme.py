import re
import subprocess
from typing import List
from typing import Tuple
from typing import Optional

from loguru import logger

# region Configuration
REGEX_SEMVER = r"(?P<currentVersion>\d+.\d+.\d+) to (?P<nextVersion>\d+.\d+.\d+)"
RE_VERSION = re.compile(REGEX_SEMVER)
DEBUG = True
README_FILE = "README.md"
START_TAG = "<!-- START BADGES -->"
END_TAG = "<!-- END BADGES -->"
# endregion

# region Functions


def run_command():
    stdout = subprocess.run(
        ["semantic-release version --noop"], shell=True, capture_output=True
    ).stderr.decode()
    return stdout


def open_readme():
    with open(README_FILE, "r") as f:
        readme = f.read()
    return readme


def write_readme(readme: str):
    with open(README_FILE, "w") as f:
        f.write(readme)


def parse_readme(readme: str) -> List[str]:
    lines = readme.split("\n")
    return lines


def get_tags_position(lines: List[str]) -> Optional[Tuple[int, int]]:
    start_position = None
    end_position = None
    for i, line in enumerate(lines):
        logger.debug(f"Line {i}\t {line}")
        if start_position is None:
            logger.debug("CHECK FOR START")
            if START_TAG in line:
                logger.success("IS START LINE")
                start_position = i
        elif end_position is None:
            logger.debug("CHECK FOR END")
            if END_TAG in line:
                logger.success("IS END LINE")
                end_position = i
        else:
            logger.success("DONE")
            break
    if start_position > end_position:
        return None, None
    return start_position, end_position


def build_release_tag(version):
    return f'<img src="https://badgen.net/badge/release/v{version}?color=blue"></img>'


def generate_readme(
    lines: List[str], tags_position: Tuple[int, int], tags=List[str]
) -> str:
    start, end = tags_position
    return "\n".join(lines[: (start + 1)] + ["".join(tags)] + lines[end:])


def main():
    msg = run_command()
    readme = open_readme()
    lines = parse_readme(readme=readme)
    start, end = get_tags_position(lines)
    if start is None or end is None:
        logger.error("No badge tags available") if DEBUG else None
        return
    logger.debug(msg) if DEBUG else None
    m = RE_VERSION.search(msg)
    if m is None:
        logger.warning("No bumps available") if DEBUG else None
        return
    next_version = m.group("nextVersion")
    tags = [build_release_tag(next_version)]
    new_readme = generate_readme(lines, (start, end), tags)
    write_readme(new_readme)


# endregion

if __name__ == "__main__":
    main()
