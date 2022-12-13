def get_f2ai_parser(parser):
    parser = parser.add_parser("initialize", help="initialize a f2ai project")
    parser.add_argument("--url", type=str, required=True, help="The project url, s3 address")


def get_materialize_parser(parser):
    parser = parser.add_parser("materialize", help="materialize a offline/online model")

    parser.add_argument("--url", type=str, required=True, help="The project url, s3 address")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--views", type=str, required=True)
    parser.add_argument("--fromnow", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--step", type=str, default="1 day")
