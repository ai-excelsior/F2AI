from argparse import ArgumentParser

from pytz import utc
from common.cmd_parser import get_f2ai_parser
from aie_feast.featurestore import FeatureStore
import pandas as pd
from datetime import datetime, timezone

TIME_COL = "event_timestamp"

if __name__ == "__main__":
    parser = ArgumentParser()
    get_f2ai_parser(parser)
    kwargs = vars(parser.parse_args())
    fs = FeatureStore(kwargs.pop("url"))
    period = "25 days"
    period = "2 days"
    entity_loan = pd.DataFrame.from_dict(
        {
            "loan": [38633],
            TIME_COL: [
                datetime(2020, 12, 12, 10, 59, 42, tzinfo=timezone.utc),
            ],
        }
    )
    entity_dobssn = pd.DataFrame.from_dict(
        {
            "dob_ssn": ["A", "B"],
            TIME_COL: [
                datetime(2022, 4, 2, tzinfo=timezone.utc),
                datetime(2021, 4, 17, 10, 59, 42, tzinfo=timezone.utc),
            ],
        }
    )

    entity_dobssn_period = pd.DataFrame.from_dict(
        {
            "dob_ssn": ["19530219_5179", "19520816_8737", "19860413_2537"],
            TIME_COL: [
                # datetime(2020, 8, 26, tzinfo=timezone.utc), 这种时间格式大小比较有bug
                datetime(2021, 8, 26, 10, 59, 42, tzinfo=timezone.utc),
                datetime(2021, 8, 26, 10, 59, 42, tzinfo=timezone.utc),
                datetime(2021, 8, 26, 10, 59, 42, tzinfo=timezone.utc),
            ],
        }
    )
    # print(fs.get_labels(fs.labels, entity_dobssn))
    # print(fs.get_features(fs.features, entity_dobssn, None))

    print(fs.get_period_labels(fs.labels, entity_dobssn_period, period, include=False))
    print(
        fs.get_period_features(fs.features, entity_dobssn_period, period, features=None, include=True)
    )  # TODO include设置再确定一下
