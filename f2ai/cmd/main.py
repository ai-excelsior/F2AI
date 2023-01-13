#! python

import pathlib
from argparse import ArgumentParser
from typing import List
from f2ai.common.cmd_parser import add_materialize_parser
from f2ai.definitions import BackOffTime
from f2ai.featurestore import FeatureStore


def materialize(url: str, services: List[str], back_off_time: BackOffTime, online: bool):
    fs = FeatureStore(url)

    for service in services:
        fs.materialize(service, back_off_time, online)


if __name__ == "__main__":
    parser = ArgumentParser(prog="F2AI")
    subparsers = parser.add_subparsers(title="subcommands", dest="commands")
    add_materialize_parser(subparsers)

    kwargs = vars(parser.parse_args())
    commands = kwargs.pop("commands", None)
    if commands == "materialize":
        if kwargs["fromnow"] is None and kwargs["start"] is None and kwargs["end"] is None:
            parser.error("One of fromnow or start&end is required.")

    if not pathlib.Path('feature_store.yml').exists():
        parser.error("No feature_store.yml found in current folder, please switch to folder which feature_store.yml exists.")

    if commands == "materialize":
        from_now = kwargs.pop("fromnow", None)
        step = kwargs.pop("step", None)

        if from_now is not None:
            back_off_time = BackOffTime.from_now(from_now=from_now, step=step)
        else:
            back_off_time = BackOffTime(start=kwargs.pop("start"), end=kwargs.pop("end"), step=step)

        materialize("file://./feature_store.yml", kwargs.pop("services"), back_off_time, kwargs.pop("online"))
