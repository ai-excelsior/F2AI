from dataclasses import dataclass, field


@dataclass
class ConnectConfig:
    """realize the connection config"""

    type: str
    host: str = field(default=None)
    port: str = field(default=None)
    database: str = field(default=None)
    schema: str = field(default=None)
    user: str = field(default=None)
    passwd: str = field(default=None)
