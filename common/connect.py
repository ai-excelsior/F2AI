from dataclasses import dataclass, field
from .utils import get_default_value


@dataclass
class ConnectConfig:
    """realize the connection config"""

    type: str
    host: str = field(default_factory=get_default_value)
    port: str = field(default_factory=get_default_value)
    database: str = field(default_factory=get_default_value)
    schema: str = field(default_factory=get_default_value)
    user: str = field(default_factory=get_default_value)
    passwd: str = field(default_factory=get_default_value)
