from pydantic import BaseModel
from typing import List, Any, Optional


class Entity(BaseModel):
    """An entity is a key connection between different feature views. Under the hook, we use join_keys to join the feature views. If join key is empty, we will take the name as a default join key"""

    name: str
    description: Optional[str]
    join_keys: List[str] = []

    def __init__(__pydantic_self__, **data: Any) -> None:

        join_keys = data.pop("join_keys", [])
        if len(join_keys) == 0:
            join_keys = [data.get("name")]

        super().__init__(**data, join_keys=join_keys)

    def __hash__(self) -> int:
        s = ",".join(self.join_keys)
        return hash(f"{self.name}:{s}")
