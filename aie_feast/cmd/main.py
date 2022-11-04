from argparse import ArgumentParser
import pandas as pd
from datetime import datetime, timezone
from aie_feast.common.cmd_parser import get_f2ai_parser
from aie_feast.featurestore import FeatureStore
from aie_feast.common.sampler import GroupFixednbrSampler, GroupRandomSampler, UniformNPerGroupSampler

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
        fs.get_latest_entities(fs.features["gy_link_travel_time_features"])

    # get_latest_entity()

    def stats():
        # entity_link = pd.DataFrame.from_dict(
        #     {
        #         #  "link": ["3377906280028510514", "4377906282959500514"],
        #         TIME_COL: [
        #             datetime(2016, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
        #             datetime(2016, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
        #         ],
        #     }
        # )
        # fs.stats(fs.features["gy_link_travel_time_features"], entity_df=entity_link, fn="mean")
        # fs.get_latest_entities(fs.features["gy_link_travel_time_features"])

        entity_dobssn_period = pd.DataFrame.from_dict(
            {
                # "dob_ssn": ["19991113_3598", "19960703_3449"],
                # "loan": ["21837", "38637"],
                TIME_COL: [
                    datetime(2021, 8, 26, tzinfo=timezone.utc),
                    datetime(2021, 1, 1, 10, 59, 42, tzinfo=timezone.utc),
                    #   datetime(2021, 7, 1, 10, 59, 42, tzinfo=timezone.utc),
                ],
            }
        )
        # fs.stats(fs.services["credit_scoring_v1"], fn="max", group_key=[], features=["event_timestamp"])
        fs.get_latest_entities(fs.services["credit_scoring_v1"])

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
                # TIME_COL: [
                #     datetime(2020, 12, 12, 10, 59, 42, tzinfo=timezone.utc),
                #     datetime(2021, 12, 12, 10, 59, 42, tzinfo=timezone.utc),
                # ],
                TIME_COL: [
                    "2021-08-21 00:41:22.000+08",
                    "2021-08-21 00:23:00.000+08",
                    "2021-08-21 00:04:39.000+08",
                    "2021-08-20 23:46:18.000+08",
                    "2021-08-20 23:27:57.000+08",
                    "2021-08-20 23:09:35.000+08",
                    "2021-08-20 22:51:14.000+08",
                    "2021-08-20 22:32:53.000+08",
                    "2021-08-20 22:14:32.000+08",
                    "2021-08-20 21:56:11.000+08",
                    "2021-08-20 21:37:49.000+08",
                    "2021-08-20 21:19:28.000+08",
                    "2021-08-20 21:01:07.000+08",
                    "2021-08-20 20:42:46.000+08",
                    "2021-08-20 20:24:24.000+08",
                    "2021-08-20 20:06:03.000+08",
                    "2021-08-20 19:47:42.000+08",
                    "2021-08-20 19:29:21.000+08",
                    "2021-08-20 19:10:59.000+08",
                    "2021-08-20 18:52:38.000+08",
                    "2021-08-20 18:34:17.000+08",
                    "2021-08-20 18:15:56.000+08",
                    "2021-08-20 17:57:35.000+08",
                    "2021-08-20 17:39:13.000+08",
                    "2021-08-20 17:20:52.000+08",
                    "2021-08-20 17:02:31.000+08",
                    "2021-08-20 16:44:10.000+08",
                    "2021-08-20 16:25:48.000+08",
                    "2021-08-20 16:07:27.000+08",
                    "2021-08-20 15:49:06.000+08",
                    "2021-08-20 15:30:45.000+08",
                    "2021-08-20 15:12:23.000+08",
                    "2021-08-20 14:54:02.000+08",
                    "2021-08-20 14:35:41.000+08",
                    "2021-08-20 14:17:20.000+08",
                    "2021-08-20 13:58:58.000+08",
                    "2021-08-20 13:40:37.000+08",
                    "2021-08-20 13:22:16.000+08",
                    "2021-08-20 13:03:55.000+08",
                    "2021-08-20 12:45:34.000+08",
                    "2021-08-20 12:27:12.000+08",
                    "2021-08-20 12:08:51.000+08",
                    "2021-08-20 11:50:30.000+08",
                    "2021-08-20 11:32:09.000+08",
                    "2021-08-20 11:13:47.000+08",
                    "2021-08-20 10:55:26.000+08",
                    "2021-08-20 10:37:05.000+08",
                    "2021-08-20 10:18:44.000+08",
                    "2021-08-20 10:00:22.000+08",
                    "2021-08-20 09:42:01.000+08",
                    "2021-08-20 09:23:40.000+08",
                    "2021-08-20 09:05:19.000+08",
                    "2021-08-20 08:46:58.000+08",
                    "2021-08-20 08:28:36.000+08",
                    "2021-08-20 08:10:15.000+08",
                    "2021-08-20 07:51:54.000+08",
                    "2021-08-20 07:33:33.000+08",
                    "2021-08-20 07:15:11.000+08",
                    "2021-08-20 06:56:50.000+08",
                    "2021-08-20 06:38:29.000+08",
                    "2021-08-20 06:20:08.000+08",
                    "2021-08-20 06:01:46.000+08",
                    "2021-08-20 05:43:25.000+08",
                    "2021-08-20 05:25:04.000+08",
                    "2021-08-20 05:06:43.000+08",
                    "2021-08-20 04:48:22.000+08",
                    "2021-08-20 04:30:00.000+08",
                    "2021-08-20 04:11:39.000+08",
                    "2021-08-20 03:53:18.000+08",
                    "2021-08-20 03:34:57.000+08",
                    "2021-08-20 03:16:35.000+08",
                    "2021-08-20 02:58:14.000+08",
                    "2021-08-20 02:39:53.000+08",
                    "2021-08-20 02:21:32.000+08",
                    "2021-08-20 02:03:10.000+08",
                    "2021-08-20 01:44:49.000+08",
                    "2021-08-20 01:26:28.000+08",
                    "2021-08-20 01:08:07.000+08",
                    "2021-08-20 00:49:45.000+08",
                    "2021-08-20 00:31:24.000+08",
                    "2021-08-20 00:13:03.000+08",
                    "2021-08-19 23:54:42.000+08",
                    "2021-08-19 23:36:21.000+08",
                    "2021-08-19 23:17:59.000+08",
                    "2021-08-19 22:59:38.000+08",
                    "2021-08-19 22:41:17.000+08",
                    "2021-08-19 22:22:56.000+08",
                    "2021-08-19 22:04:34.000+08",
                    "2021-08-19 21:46:13.000+08",
                    "2021-08-19 21:27:52.000+08",
                    "2021-08-19 21:09:31.000+08",
                    "2021-08-19 20:51:09.000+08",
                    "2021-08-19 20:32:48.000+08",
                    "2021-08-19 20:14:27.000+08",
                    "2021-08-19 19:56:06.000+08",
                    "2021-08-19 19:37:45.000+08",
                    "2021-08-19 19:19:23.000+08",
                    "2021-08-19 19:01:02.000+08",
                    "2021-08-19 18:42:41.000+08",
                    "2021-08-19 18:24:20.000+08",
                    "2021-08-19 18:05:58.000+08",
                    "2021-08-19 17:47:37.000+08",
                    "2021-08-19 17:29:16.000+08",
                    "2021-08-19 17:10:55.000+08",
                    "2021-08-19 16:52:33.000+08",
                    "2021-08-19 16:34:12.000+08",
                    "2021-08-19 16:15:51.000+08",
                    "2021-08-19 15:57:30.000+08",
                    "2021-08-19 15:39:09.000+08",
                    "2021-08-19 15:20:47.000+08",
                    "2021-08-19 15:02:26.000+08",
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

        fs.get_labels("loan_label_view", entity_loan, "365 days")
        # fs.get_period_features(fs.features["gy_link_travel_time_features"], entity_link, period="5 hours")
        # fs.get_features(fs.features["zipcode_features"], entity_loan)
        # fs.get_labels(fs.service["credit_scoring_v1"], entity_dobssn_period)

    get_features()

    def do_materailize():
        # fs.materialize("traval_time_prediction_embedding_v1", incremental_begin="2016-06-30 07:50:00")
        # fs.materialize("traval_time_prediction_embedding_v1")
        fs.materialize("traval_time_prediction_embedding_v1", incremental_begin="3 minutes")

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
        # ds = fs.get_dataset(
        #     service_name="credit_scoring_v1",
        #     sampler=GroupFixednbrSampler(
        #         time_bucket="10 days",
        #         stride=2,
        #         group_ids=groups,
        #         group_names=["zipcode", "dob_ssn"],
        #         start="2020-12-24",
        #         end="2021-08-26",
        #     ),
        # )
        # groups = fs.stats(
        #     fs.feature_views["gy_link_travel_time_features"],
        #     group_key=["link"],
        #     keys_only=True,
        #     fn="unique",
        #     start="2016-03-01 00:02:00",
        #     end="2016-06-30 08:00:00",
        # )
        groups = fs.stats(
            fs.feature_views["loan_features"],
            group_key=["loan", "dob_ssn"],
            #    fs.feature_views["loan_features"],
            # group_key=["zipcode", "dob_ssn"],
            keys_only=True,
            fn="unique",
            start="2021-08-24",
            end="2021-08-26",
        )
        ds = fs.get_dataset(
            service_name="credit_scoring_v1",
            sampler=GroupFixednbrSampler(
                time_bucket="10 days",
                stride=1,
                group_ids=groups,
                group_names=["loan", "dob_ssn"],
                start="2020-08-01",
                end="2021-09-30",
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
