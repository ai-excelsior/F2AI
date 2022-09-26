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
<<<<<<< HEAD
<<<<<<< HEAD
    period = "25 days"
=======
    period = "2 days"
>>>>>>> 8a68f53 (get_period_features/labels)
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
    print(fs.get_labels(fs.labels, entity_dobssn))
    print(fs.get_features(fs.features, entity_dobssn, None))

<<<<<<< HEAD
    # print(fs.get_period_labels(fs.labels, entity_dobssn, period, include=False))
    # print(fs.get_period_features(fs.labels, entity_dobssn, period, features=None, include=False))
=======

    def features_and_labels():
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
                "dob_ssn": ["A", "A", "A", "A", "B", "B", "B", "B"],
                TIME_COL: [
                    datetime(2022, 4, 2, tzinfo=timezone.utc),
                    datetime(2021, 4, 17, 10, 59, 42, tzinfo=timezone.utc),
                ],
            }
        )
        print(fs.get_labels(fs.labels, entity_dobssn))
        print(fs.get_features(fs.features, entity_dobssn, None))

    def entities():
        entity_fea = fs.get_latest_entities(fs.features["credit_history_features"])
        entity_lab = fs.get_latest_entities(fs.labels)
        entity_all = fs.get_latest_entities(fs.features + fs.labels)

    entities()
>>>>>>> 61f7491 (update get_latest_entities)
=======
    # print(fs.get_period_labels(fs.labels, entity_dobssn_period, period, include=False))
    # print(
    #     fs.get_period_features(fs.features, entity_dobssn_period, period, features=None, include=False)
    # )  # TODO include设置再确定一下
>>>>>>> 8a68f53 (get_period_features/labels)
