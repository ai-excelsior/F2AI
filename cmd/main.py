from argparse import ArgumentParser
import os
from pytz import utc
from common.cmd_parser import get_f2ai_parser
from aie_feast.featurestore import FeatureStore
import pandas as pd
from datetime import datetime, timezone
from common.sampler import GroupFixednbrSampler, GroupRandomSampler, UniformNPerGroupSampler

TIME_COL = "event_timestamp"

if __name__ == "__main__":
    parser = ArgumentParser()
    get_f2ai_parser(parser)
    kwargs = vars(parser.parse_args())
    fs = FeatureStore(kwargs.pop("url"))

    # fs.get_latest_entities(fs.features)

    def get_period_data(period):
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
        # print(fs.get_features(fs.features, entity_dobssn, None))

        print(fs.get_period_labels(fs.labels, entity_dobssn_period, period, include=False))
        print(
            fs.get_period_features(fs.features, entity_dobssn_period, period, features=None, include=True)
        )  # TODO include设置再确定一下

    def get_latest_entity():
        fs.get_latest_entities(fs.features["loan_features"])

    def stats():
        entity_link = pd.DataFrame.from_dict(
            {
                "link": ["3377906280028510514", "4377906282959500514"],
                TIME_COL: [
                    datetime(2016, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
                    datetime(2016, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
                ],
            }
        )
        fs.stats(fs.features["gy_link_travel_time_features"], group_key=["link"], fn="mean")
        fs.get_latest_entities(fs.features["gy_link_travel_time_features"])

    # stats()

    def get_features():
        entity_link = pd.DataFrame.from_dict(
            {
                #  "link": ["3377906280028510514"],
                TIME_COL: [
                    datetime(2016, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
                ],
            }
        )
        entity_loan = pd.DataFrame.from_dict(
            {
                # "loan": ["38633", "38633"],
                TIME_COL: [
                    datetime(2020, 12, 12, 10, 59, 42, tzinfo=timezone.utc),
                    datetime(2021, 12, 12, 10, 59, 42, tzinfo=timezone.utc),
                ],
            }
        )
        entity_dobssn_period = pd.DataFrame.from_dict(
            {
                "dob_ssn": ["19991113_3598", "19960703_3449"],
                "loan": ["21837", "38637"],
                TIME_COL: [
                    datetime(2021, 8, 26, tzinfo=timezone.utc),
                    datetime(2021, 1, 1, 10, 59, 42, tzinfo=timezone.utc),
                    #   datetime(2021, 7, 1, 10, 59, 42, tzinfo=timezone.utc),
                ],
            }
        )  # 19991113_3598 has duplicates, due to the original data

        fs.get_labels(fs.service["credit_scoring_v1"], entity_dobssn_period, "365 days")
        # fs.get_period_features(fs.features["gy_link_travel_time_features"], entity_link, period="5 hours")
        # fs.get_features(fs.features["zipcode_features"], entity_loan)
        # fs.get_labels(fs.service["credit_scoring_v1"], entity_dobssn_period)

    get_features()

    def do_materailize():
        fs.materialize(
            "credit_scoring_v1",
        )

    # do_materailize()

    def get_period_features_and_labels():

        period = "5 hours"
        entity_link_ID_period = pd.DataFrame.from_dict(
            {
                "link": [
                    "3377906289228510514",
                ],
                TIME_COL: [
                    datetime(2016, 5, 30, 0, 0, 0, tzinfo=timezone.utc),
                ],
            }
        )
        entity_dobssn_period = pd.DataFrame.from_dict(
            {
                "dob_ssn": ["19991113_3598", "19960703_3449"],
                "loan": ["21837", "38637"],
                TIME_COL: [
                    datetime(2021, 8, 26, tzinfo=timezone.utc),
                    datetime(2021, 1, 1, 10, 59, 42, tzinfo=timezone.utc),
                    #   datetime(2021, 7, 1, 10, 59, 42, tzinfo=timezone.utc),
                ],
            }
        )
        fs.get_period_features(
            fs.features["gy_link_travel_time_features"], entity_link_ID_period, period, include=False
        )
        fs.get_period_labels(
            fs.labels["travel_time_label_view"], entity_link_ID_period, period, include=False
        )

    # get_period_features_and_labels()

    def dataset():
        groups = fs.stats(
            fs.features["loan_features"],
            group_key=["zipcode", "dob_ssn"],
            keys_only=True,
            fn="unique",
            start="2021-08-24",
            end="2021-08-26",
        )
        ds = fs.get_dataset(
            service_name="credit_scoring_v1",
            sampler=GroupFixednbrSampler(
                time_bucket="10 days",
                stride=2,
                group_ids=groups,
                group_names=["zipcode", "dob_ssn"],
                start="2020-12-24",
                end="2021-08-26",
            ),
        )
        i_ds = ds.to_pytorch()
        next(iter(i_ds))

    # dataset()

    def sample():
        time_bucket = "4 days"
        stride = 3
        start = "2010-01-01 00:00:00"
        end = "2010-01-30 00:00:00"
        group_ids = [("A", 10), ("A", 11), ("B", 10), ("B", 11)]
        # group_ids = (("A", 10), ("A", 11), ("B", 10), ("B", 11))
        # group_ids = ("A", "B")
        # group_ids = ["A", "B"]

        ratio = 0.5
        n_groups = 2
        avg_nbr = 2

        # sample1 = GroupFixednbrSampler(time_bucket, stride, start, end)()
        # sample2 = GroupRandomSampler(time_bucket, stride, ratio, start, end)()
        # sample3 = UniformNPerGroupSampler(time_bucket, stride, n_groups, avg_nbr, start, end)()
        sample1 = GroupFixednbrSampler(
            time_bucket, stride, start, end, group_ids=group_ids, group_names=["LETTER", "NBR"]
        )()
        sample2 = GroupRandomSampler(
            time_bucket, stride, ratio, start, end, group_ids=group_ids, group_names=["LETTER", "NBR"]
        )()
        sample3 = UniformNPerGroupSampler(
            time_bucket, stride, n_groups, avg_nbr, start, end, group_ids, group_names=["LETTER", "NBR"]
        )()
        print(sample1)
        print(sample2)
        print(sample3)

    # sample()
