import re
from enum import Enum

# from loguru import logger
from semantic_release import ImproperConfigurationError
from semantic_release import UnknownCommitMessageStyleError
from semantic_release.history.parser_helpers import ParsedCommit
from semantic_release.history.parser_helpers import re_breaking
from semantic_release.history.parser_helpers import parse_paragraphs


class CommitTags(Enum):
    ADD = "[ADD]"
    BREAK = "[BREAK]"
    BREAKING = "[BREAKING]"
    CI = "[CI]"
    CONFIG = "[CONFIG]"
    DEBUG = "[DEBUG]"
    DEV = "[DEV]"
    DOC = "[DOC]"
    FEAT = "[FEAT]"
    FIX = "[FIX]"
    REFACTO = "[REFACTO]"
    RM = "[RM]"
    TEST = "[TEST]"
    STYLE = "[STYLE]"


LONG_TYPE_NAMES = {
    "[ADD]": "diff",
    "[BREAK]": "breaking",
    "[BREAKING]": "breaking",
    "[CI]": "CICD",
    "[CONFIG]": "configuration",
    "[DEBUG]": "diff",
    "[DEV]": "diff",
    "[DOC]": "documentation",
    "[FEAT]": "feature",
    "[FIX]": "fix",
    "[REFACTO]": "refactor",
    "[RM]": "diff",
    "[TEST]": "test",
    "[STYLE]": "style",
}


class OcarinowParserConfig:

    LEVEL_BUMPS = {"no-release": 0, "patch": 1, "minor": 2, "major": 3}
    TAGS_FOR_REGEX = [
        tag.value.replace("[", "\[").replace("]", "\]") for tag in CommitTags
    ]
    NONE_TAGS = {
        CommitTags.ADD,
        CommitTags.CI,
        CommitTags.CONFIG,
        CommitTags.DEBUG,
        CommitTags.DEV,
        CommitTags.DOC,
        CommitTags.REFACTO,
        CommitTags.RM,
        CommitTags.STYLE,
        CommitTags.TEST,
    }
    PATCH_TAGS = {CommitTags.TEST, CommitTags.FIX}
    MINOR_TAGS = {CommitTags.FEAT}
    MAJOR_TAGS = {CommitTags.BREAK, CommitTags.BREAKING}


def ocarinow_commit_parser(msg: str) -> ParsedCommit:
    """Parse Commit Message for Ocarinow Projects

    Args:
        msg(str): Commit message

    Returns:
        (ParsedCommit):

    Notes:
        n/a

    Tests:
        n/a

    Raises:
        (UnknownCommitMessageStyleError): The commit message was not parsable
    """
    # First Match ALL Tags from the Commit
    re_parser = re.compile(
        r"(?P<type>" + "|".join(OcarinowParserConfig.TAGS_FOR_REGEX) + ")+"
        r"(?:\((?P<scope>[^\n]+)\))?"
        r"(?P<break>!)? "
        r"(?P<subject>[^\n]+)"
        r"(:?\n\n(?P<text>.+))?",
        re.DOTALL,
    )

    parsed = re_parser.match(msg)
    if not parsed:
        raise UnknownCommitMessageStyleError(
            f"Unable to parse the given commit message: {msg}"
        )

    # We get all the matching groups
    parsed_break = parsed.group("break")
    parsed_scope = parsed.group("scope")
    parsed_subject = parsed.group("subject")
    parsed_text = parsed.group("text")
    parsed_type = parsed.group("type")

    # parsed_commit = parsed_message[0]
    if parsed_text:
        descriptions = parse_paragraphs(parsed_text)
    else:
        descriptions = list()
    descriptions.insert(0, parsed_subject)

    # Look for descriptions of breaking changes
    breaking_descriptions = [
        match.group(1)
        for match in (re_breaking.match(p) for p in descriptions[1:])
        if match
    ]

    default_level_bump = "no-release"
    if default_level_bump not in OcarinowParserConfig.LEVEL_BUMPS.keys():
        raise ImproperConfigurationError(
            f"{default_level_bump} is not a valid option for "
            f"parser_ocarinow_default_level_bump.\n"
            f"valid options are: {', '.join(OcarinowParserConfig.LEVEL_BUMPS.keys())}"
        )
    level_bump = OcarinowParserConfig.LEVEL_BUMPS[default_level_bump]

    try:
        commit_tag = CommitTags(parsed_type)
    except ValueError:
        # logger.error(e)
        raise ValueError(f"{parsed_type} is not a Valid TAG for Ocarinow SemVer")

    if (
        commit_tag in OcarinowParserConfig.MAJOR_TAGS
        or parsed_break
        or breaking_descriptions
    ):
        level_bump = 3
    elif commit_tag in OcarinowParserConfig.MINOR_TAGS:
        level_bump = 2
    elif commit_tag in OcarinowParserConfig.PATCH_TAGS:
        level_bump = 1
    elif commit_tag in OcarinowParserConfig.NONE_TAGS:
        level_bump = 0

    parsed_type_long = LONG_TYPE_NAMES.get(parsed_type, parsed_type)

    return ParsedCommit(
        level_bump,
        parsed_type_long,
        parsed_scope,
        descriptions,
        breaking_descriptions,
    )


MSG = """[FEAT][CONFIG] Add a breaking change to the codebase

Here we want to display a list:
- This is a first point
- This is a second point

BREAKING CHANGE: The function funcA is not working anymore

BREAKING CHANGE: The function funcB use a new argument"""
if __name__ == "__main__":

    result = ocarinow_commit_parser(MSG)
    print(result)
