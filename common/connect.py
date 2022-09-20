from dataclasses import dataclass, field


@dataclass
class ConnectConfig:
    """realize the connection config"""

    type: str
    host: str = field(default_factory=str)
    port: int = field(default_factory=int)
    database: str = field(default_factory=str)
    schema: str = field(default_factory=str)
    user: str = field(default_factory=str)
    passwd: str = field(default_factory=str)
