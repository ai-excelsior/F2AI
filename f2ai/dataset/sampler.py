from typing import List


class AbstractSampler:
    def __init__(self, time_bucket: str = "1 day", random_seed: int = 666, stride: int = 1):
        self.time_bucket = time_bucket
        self.random_seed = random_seed
        self.stride = stride

    def __call__(self, n: int, keys: List[str]):
        return list(range(n))


class RandomSampler(AbstractSampler):
    """random sample based on the time-bucket

    Args:
        AbstractSampler (_type_): _description_
    """

    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio


class GroupRandomSampler(AbstractSampler):
    """random sample based on the time-bucket

    Args:
        AbstractSampler (_type_): _description_
    """

    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio


class FixednbrSampler(AbstractSampler):
    """uniformly sample `ratio` samples according to time-bucket, dis-regard groups

    Args:
        AbstractSampler (_type_): _description_
    """

    def __init__(self, nbr=10):
        super().__init__()
        self.nbr = nbr


class GroupFixednbrSampler(AbstractSampler):
    """uniformly sample `ratio` samples according to time-bucket, dis-regard groups

    Args:
        AbstractSampler (_type_): _description_
    """

    def __init__(self, nbr=10):
        super().__init__()
        self.nbr = nbr


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
