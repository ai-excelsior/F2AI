import datetime
from enum import Enum

LOCAL_TIMEZONE = datetime.datetime.now().astimezone().tzinfo


class StatsFunctions(Enum):
    MIN = "min"
    MAX = "max"
    STD = "std"
    AVG = "avg"
    MODE = "mode"
    MEDIAN = "median"
    UNIQUE = "unique"
