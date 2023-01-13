from argparse import ArgumentParser


def add_materialize_parser(subparsers):
    parser: ArgumentParser = subparsers.add_parser(
        "materialize", help="materialize a service to offline (default) or online"
    )

    parser.add_argument("services", type=str, nargs='+', help="at least one service name, multi service name using space to separate.")
    parser.add_argument(
        "--online", action="store_true", help="materialize service to online store if presents."
    )
    parser.add_argument("--fromnow", type=str, help="materialize start time point from now, egg: 7 days.")
    parser.add_argument(
        "--start", type=str, help="materialize start time point, egg: 2022-10-22, or 2022-11-22T10:12."
    )
    parser.add_argument(
        "--end", type=str, help="materialize end time point, egg: 2022-10-22, or 2022-11-22T10:12."
    )
    parser.add_argument("--step", type=str, default="1 day", help="how to split materialize task")
