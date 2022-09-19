import pandas as pd
from datetime import datetime

entity_df = pd.DataFrame.from_dict(
    {
        # 实体ID
        "payment_method": ["credit_card", "gift_card"],
        # 时间戳，查询小于该时间点的作为上下文
        "event_timestamp": [
            datetime(2021, 4, 12, 10, 59, 42),
            datetime(2021, 4, 12, 8, 12, 10),
        ],
    }
)
