from f2ai.definitions import BackOffTime


def test_back_off_time_to_units():
    back_off_time = BackOffTime(start="2020-08-01 12:20", end="2020-10-30", step="1 month")
    units = list(back_off_time.to_units())
    assert len(units) == 3

    back_off_time = BackOffTime(start="2020-08-01 12:20", end="2020-08-02", step="2 hours")
    units = list(back_off_time.to_units())
    assert len(units) == 5
