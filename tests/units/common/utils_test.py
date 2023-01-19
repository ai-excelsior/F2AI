from f2ai.common.utils import batched


def test_batched():
    xs = range(10)
    batches = next(batched(xs, batch_size=3))
    assert batches == [0, 1, 2]

    last_batch = []
    for batch in batched(xs, batch_size=3):
        last_batch = batch
    assert last_batch == [9]
