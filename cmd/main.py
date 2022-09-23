from argparse import ArgumentParser

from pytz import utc
from common.cmd_parser import get_f2ai_parser
from aie_feast.featurestore import FeatureStore
import pandas as pd
from datetime import datetime, timezone

if __name__ == "__main__":
    parser = ArgumentParser()
    get_f2ai_parser(parser)
    kwargs = vars(parser.parse_args())
    fs = FeatureStore(kwargs.pop("url"))
    entity_loan = pd.DataFrame.from_dict(
        {
            "loan": [38633],
            "event_timestamp": [
                datetime(2020, 12, 12, 10, 59, 42, tzinfo=timezone.utc),
            ],
        }
    )
    entity_dobssn = pd.DataFrame.from_dict(
        {
            "dob_ssn": ["19621030_8837", "19831011_2467"],
            "event_timestamp": [
                datetime(2021, 8, 23, 10, 59, 42, tzinfo=timezone.utc),
                datetime(2021, 8, 17, 10, 59, 42, tzinfo=timezone.utc),
            ],
        }
    )
    print(fs.get_features(fs.features, entity_dobssn, None))
