from typing import Dict, Any, List, Union


class AbstractSampler:
    def __init__(self, time_bucket: str = "1day"):
        self.time_bucket = time_bucket

    def __call__(self, n: int, keys: List[str]):
        return list(range(n))


class UniformSampler(AbstractSampler):
    """uniformly sample `ratio` samples according to time-bucket, dis-regard groups

    Args:
        AbstractSampler (_type_): _description_
    """

    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio


class UniformNPerGroupSampler(AbstractSampler):
    """sample `avg_nbr` for each group

    Args:
        AbstractSampler (_type_): _description_
    """

    def __init__(self, avg_nbr: int, n_groups: int = 0):
        super().__init__()
        self.avg_nbr = avg_nbr
        self.n_groups = n_groups


class ScaleHistogramSampler(AbstractSampler):
    pass
