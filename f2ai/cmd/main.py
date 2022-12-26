from argparse import ArgumentParser
from f2ai.common.cmd_parser import get_f2ai_parser, get_materialize_parser
from f2ai.definitions.backoff_time import cfg_to_date
from f2ai.featurestore import FeatureStore

TIME_COL = "event_timestamp"


def init(url):
    return FeatureStore(url)


def materialize(url, views, backoff, online):
    fs = FeatureStore(url)

    for view in views:
        fs.materialize(view, backoff, online)
        # fs.get_online_features(view, entity_df=pd.DataFrame(data=[1, 2, 3], columns=["loan_id"]))

    # for view in views:
    #     try:
    #         fs.materialize(view, backoff, online)
    #         print(f"{view} materialize done")
    #     except:
    #         print(
    #             f"{view} materialize failed, please check whether view can be materialized in online={online} mode"
    #         )


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    init_parser = get_f2ai_parser(subparsers)

    get_materialize_parser(subparsers)
    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")
    import pandas as pd

    entity_df = pd.DataFrame(["43938", "46990", "72202"], columns=["zipcode"])

    entity_df_loan_features = pd.DataFrame(
        [["19650216_4059", "43938"], ["19730313_4796", "46990"], ["19880427_8332", "72202"]],
        columns=["dob_ssn", "zipcode"],
    )

    entity_df_history = pd.DataFrame(["19650216_4059", "19730313_4796"], columns=["dob_ssn"])
    if command == "initialize":
        fs = init(kwargs.pop("url"))
        fs.get_online_features("credit_scoring_v1", entity_df_loan_features)
    elif command == "materialize":
        materialize_time = cfg_to_date(
            kwargs.pop("fromnow"), kwargs.pop("start"), kwargs.pop("end"), kwargs.pop("step")
        )
        materialize(kwargs.pop("url"), kwargs.pop("views").split(","), materialize_time, kwargs.pop("online"))
