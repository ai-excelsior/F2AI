from enum import Enum


class FeatureDTypes(str, Enum):
    """Feature data type definitions which supported by F2AI. Used to convert to a certain data type for different algorithm frameworks. Not useful now, but future."""

    INT = "int"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT = "float"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BOOLEAN = "bool"
    UNKNOWN = "unknown"


NUMERIC_FEATURE_D_TYPES = {
    FeatureDTypes.INT,
    FeatureDTypes.INT32,
    FeatureDTypes.INT64,
    FeatureDTypes.FLOAT,
    FeatureDTypes.FLOAT32,
    FeatureDTypes.FLOAT64,
}
