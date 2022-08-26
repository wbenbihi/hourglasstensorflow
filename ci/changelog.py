import re
import subprocess
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Literal
from typing import Callable
from typing import Optional
from datetime import datetime

import dateparser
from loguru import logger
from pydantic import Field
from pydantic import BaseModel

from ci.commit_parser import ParsedCommit
from ci.commit_parser import UnknownCommitMessageStyleError
from ci.commit_parser import ocarinow_commit_parser

# region CONFIGURATION
## Utility Files
# LOG_FILE = "scripts/data.ignore.txt"
MARKDOWN_FILE = "CHANGELOG.ignore.md"
DEBUG = False
## Regex
RE_COMMIT = re.compile(
    r"commit (?P<commitId>\w*) ?"
    r"\(?(?P<commitInfo>[^\)\n]*)?\)?"
    r"(\nMerge: (?P<mergeInfo>[^\n]+))?"
    r"(\nAuthor: (?P<author>[^\)\n]+))?"
    r"(\nDate:   (?P<date>[^\)\n]+))"
    r"\n\n(?P<msg>(.|\n)*)"
)
RE_VERSION = re.compile(r"^(?P<version>\d.\d.\d)\n\n")
RE_TAG_VERSION = re.compile(r"tag: (?P<version>v\d+.\d+.\d+)")
## Default Configuration
REPO_URL = "https://github.com/wbenbihi/hourglasstensorflow"
BREAKING_SUFFIX = "BREAKING"
CURRENT_VERSION = "__CURRENT__"
GROUP_KEYS = [
    "commitId",
    "commitInfo",
    "mergeInfo",
    "author",
    "date",
    "msg",
]

COMMIT_TYPE_ORDER = [
    "breaking",
    "feature",
    "fix",
    "configuration",
    "documentation",
    "CICD",
    "test",
    "refactor",
    "style",
    "diff",
]

# endregion

# region Base Commit Class


class BaseCommit(BaseModel):
    commitId: str
    commitInfo: str
    mergeInfo: Optional[str] = None
    author: str
    date: str
    msg: str
    message_cleaned: bool = False
    bumpVersion: Optional[str] = None
    commitDetail: Optional[ParsedCommit] = None

    class Config:
        arbitrary_types_allowed = True

    def get_hash(self):
        return self.commitId[:7]

    def clean_message(self):
        self.msg = self.msg.replace("    ", "")
        self.message_cleaned = True
        return self

    def parse_commit_message(self):
        if not self.message_cleaned:
            raise ValueError("Commit Message is not clean")
        try:
            self.commitDetail = ocarinow_commit_parser(self.msg)
        except UnknownCommitMessageStyleError as e:
            if DEBUG:
                logger.exception(e)
        try:
            m = RE_TAG_VERSION.search(self.commitInfo)
            # m = RE_VERSION.match(self.msg)
            if m is None:
                raise ValueError("No bumpVersion")
            # self.bumpVersion = f"v{m.group('version')}"
            self.bumpVersion = m.group("version")
        except ValueError as e:
            # print("========================")
            # print("ERROR", self)
            # print("========================")
            if DEBUG:
                logger.exception(e)
        return self


class VersionCommits(BaseModel):
    version: str
    order: int
    items: List[BaseCommit] = Field(default_factory=list)
    types: Dict[str, List[int]] = Field(default_factory=dict)


# endregion

# region ChangelogBuilder

LineTypes = Union[
    Literal["LINE"],
    Literal["TITLE"],
    Literal["SUBTITLE"],
    Literal["SECTION"],
    Literal["NEWLINE"],
    Literal["ITEM"],
    Literal["SUBITEM"],
]

render_mapping: Dict[str, Callable[[str], str]] = {
    "LINE": lambda x: x,
    "TITLE": lambda x: f"# {x}",
    "SUBTITLE": lambda x: f"## {x}",
    "SECTION": lambda x: f"### {x}",
    "NEWLINE": lambda x: "",
    "ITEM": lambda x: f"* {x}",
    "SUBITEM": lambda x: f"  * {x}",
}


class LineModel(BaseModel):
    type: LineTypes
    text: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)


class ChangelogBuilder(BaseModel):
    lines: List[LineModel] = Field(default_factory=list)

    def render(self):
        rendered_lines = [self._render_line(line) for line in self.lines]
        return "\n".join(rendered_lines)

    def _prepare_message(self, msg: str, **kwargs) -> str:
        rendered_message = msg
        string_transformation = kwargs.get("transform")
        date: datetime = kwargs.get("date")
        if string_transformation == "capitalize":
            rendered_message = rendered_message.capitalize()
        if string_transformation == "upper":
            rendered_message = rendered_message.upper()
        if string_transformation == "lower":
            rendered_message = rendered_message.lower()
        if date:
            rendered_message += f" ({date.strftime('%Y-%m-%d')})"
        return rendered_message

    def _render_line(self, line: LineModel) -> str:
        rendered_message = self._prepare_message(msg=line.text, **line.args)
        return render_mapping[line.type](rendered_message)

    def _add_line(self, line: Union[LineModel, Dict]):
        self.lines.append(LineModel.parse_obj(line) if isinstance(line, dict) else line)
        return self

    def add_line(self, msg, **kwargs):
        return self._add_line({**dict(type="LINE", args=kwargs), **dict(text=msg)})

    def add_newline(self):
        return self._add_line(dict(type="NEWLINE", text=""))

    def add_title(self, msg, **kwargs):
        return self._add_line(
            {
                **dict(type="TITLE", args={**dict(transform="capitalize"), **kwargs}),
                **dict(text=msg),
            }
        )

    def add_subtitle(self, msg, **kwargs):
        return self._add_line(
            {
                **dict(
                    type="SUBTITLE", args={**dict(transform="capitalize"), **kwargs}
                ),
                **dict(text=msg),
            }
        )

    def add_section(self, msg, **kwargs):
        return self._add_line(
            {
                **dict(type="SECTION", args={**dict(transform="capitalize"), **kwargs}),
                **dict(text=msg),
            }
        )

    def add_item(self, msg, **kwargs):
        return self._add_line({**dict(type="ITEM", args=kwargs), **dict(text=msg)})

    def add_subitem(self, msg, **kwargs):
        return self._add_line({**dict(type="SUBITEM", args=kwargs), **dict(text=msg)})

    def build_header(self) -> None:
        # Construct Header
        self.add_title("Changelog")
        self.add_line(
            "All notable changes to this project will be documented in this file"
        )
        self.add_newline()
        self.add_line("<!--next-version-placeholder-->")
        self.add_newline()

    def build_versions(self, commit_dict: Dict[str, VersionCommits]) -> None:
        # Construct Version Changelog
        # Sort Version
        tagged_versions = [
            v for v in commit_dict.values() if v.version != CURRENT_VERSION
        ]
        tagged_versions.sort(
            key=lambda v: [int(d) for d in v.version.replace("v", "").split(".")],
            reverse=True,
        )
        # Remove the commits from current version
        current_version = [
            v for v in commit_dict.values() if v.version == CURRENT_VERSION
        ]

        sorted_versions = current_version + tagged_versions
        for version_commit in sorted_versions:
            # Get Active Version
            active_version = version_commit.version
            if not version_commit.items:
                if DEBUG:
                    logger.debug(f"BREAK NO ITEMS {version_commit.version}")
                # Pass current version_commit if no commits available
                break
            if active_version == CURRENT_VERSION:
                # TODO: For not versioned commits
                pass
            version_date = dateparser.parse(version_commit.items[0].date)
            # Add Version Subtitle
            self.add_subtitle(active_version, date=version_date)
            self.add_newline()

            # Build Sections
            self.build_type_section(version_commit)

    def build_type_section(self, version_commit: VersionCommits) -> None:
        for section_type in COMMIT_TYPE_ORDER:
            section_indexes = version_commit.types.get(section_type)
            if section_indexes is None:
                # If current section type not present in version
                # break the current iteration
                if DEBUG:
                    logger.debug(
                        f"BREAK NO SCOPE {version_commit.version} {section_type} {section_indexes}"
                    )
                continue
            # Loop over commits
            if section_type.upper() == "CICD":
                self.add_section(section_type, transform="upper")
            else:
                self.add_section(section_type)
            self.add_newline()
            for index in section_indexes:
                commit = version_commit.items[index]
                self.build_commit(commit)
            self.add_newline()

    def build_commit(self, commit: BaseCommit) -> None:
        # Get Details
        commitDetail = commit.commitDetail
        commitId = commit.commitId
        commitHash = commit.get_hash()
        commitScope = commitDetail.scope
        commitDescriptions: List[str] = commitDetail.descriptions
        commitMainMsg = commitDescriptions[0]
        commitBrDescriptions: List[str] = commitDetail.breaking_descriptions
        # Format main commit message
        scope_string = f"**{commitScope}** " if commitScope else ""
        message = f"{scope_string}{commitMainMsg} ([`{commitHash}`]({REPO_URL}/commit/{commitId}))"
        self.add_item(message)
        for additional_msg in commitDescriptions[1:]:
            if not additional_msg.startswith(BREAKING_SUFFIX):
                self.add_subitem(additional_msg)
        for breaking_msg in commitBrDescriptions:
            self.add_subitem(f"**BREAKING** {breaking_msg}")


# endregion

# region Step by Step Functions


def open_file() -> List[str]:
    # # Read LOG File
    # with open(LOG_FILE, "r") as f:
    #     logs = f.read()

    logs = subprocess.run(
        ["git --no-pager log --decorate"], shell=True, stdout=subprocess.PIPE
    ).stdout.decode()

    # Split Each Line
    lines = logs.split("\n")
    return lines


def get_groups(m):
    return {key: m.group(key) for key in GROUP_KEYS}


def group_commits_strings(lines: List[str]) -> List[str]:
    # Group Commit paragraph
    commits = []
    current_item = []
    for line in lines:
        # We check if this line is a new commit
        if line.startswith("commit "):
            if current_item:
                # If we enter a new commit we push the current item
                commits.append(current_item)
            # We initialize a new commit
            current_item = []
        # We populate the item
        current_item.append(line)

    # We rebuild each commit as a String
    commits: List[str] = ["\n".join(commit) for commit in commits]

    return commits


def parse_commits(commits: List[str]) -> List[BaseCommit]:
    # We extract the commit info with a regex
    parsed_commits = [get_groups(RE_COMMIT.search(commit)) for commit in commits]
    # We generate object from the parsed commits
    # We then clean the object and parsed the commit message
    commit_objs = [
        BaseCommit.parse_obj(commit).clean_message().parse_commit_message()
        for commit in parsed_commits
    ]
    return commit_objs


def split_commit_by_version(commit_objs: List[BaseCommit]) -> Dict[str, VersionCommits]:
    # We create a commit_trace
    # to group BaseCommit with same version inside VersionCommits
    version = CURRENT_VERSION
    order = 0
    commit_trace: Dict[str, VersionCommits] = {}
    for commit in commit_objs:
        if commit.bumpVersion is not None:
            # When a commit has a bump version we increment the order
            version = f"{commit.bumpVersion}"
            order += 1
        if version not in commit_trace:
            # If the version has never been seen before
            # We create a new VersionCommits
            commit_trace[version] = VersionCommits(version=version, order=order)
        # Continue to populate the current VersionCommit
        commit_trace[version].items.append(commit)
    return commit_trace


def identify_commit_types(commit_trace: Dict[str, VersionCommits]) -> None:
    # Within each VersionCommits' BaseCommit,
    # we want to split commits by commitDetail.type
    for version, version_commits in commit_trace.items():
        for i, commit in enumerate(version_commits.items):
            commitType = commit.commitDetail.type if commit.commitDetail else None
            if commitType is not None:
                if commitType not in version_commits.types:
                    version_commits.types[commitType] = []
                version_commits.types[commitType].append(i)


def build_changelog(commit_trace: Dict[str, VersionCommits]) -> ChangelogBuilder:
    builder = ChangelogBuilder()
    builder.build_header()
    builder.build_versions(commit_trace)
    return builder


# endregion


def main():
    lines = open_file()
    commits = group_commits_strings(lines)
    commit_objs = parse_commits(commits)
    commit_trace = split_commit_by_version(commit_objs)
    identify_commit_types(commit_trace)
    builder = build_changelog(commit_trace)

    changelog = builder.render()

    with open(MARKDOWN_FILE, "w") as f:
        f.write(changelog)


if __name__ == "__main__":
    lines = open_file()
    commits = group_commits_strings(lines)
    commit_objs = parse_commits(commits)
    commit_trace = split_commit_by_version(commit_objs)
    identify_commit_types(commit_trace)
    builder = build_changelog(commit_trace)

    changelog = builder.render()

    with open(MARKDOWN_FILE, "w") as f:
        f.write(changelog)
