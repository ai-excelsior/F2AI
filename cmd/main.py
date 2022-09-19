from argparse import ArgumentParser
from common.cmd_parser import get_f2ai_parser
from aie_feast.featurestore import FeatureStore

if __name__ == "__main__":
    parser = ArgumentParser()
    get_f2ai_parser(parser)
    kwargs = vars(parser.parse_args())
    FeatureStore(kwargs.pop("url"))
