from raspberry_pi_congestion.window_aggregator import WindowAggregator


class Clock:
    value = 0.0
    def __call__(self): return self.value


def test_five_second_window_average_peak_and_sample_count():
    clock = Clock()
    agg = WindowAggregator(5, clock)
    for count, now in [(2, 1000), (4, 2000), (6, 3000)]: agg.add_sample(count, now)
    clock.value = 5
    assert agg.should_flush()
    result = agg.flush(6000)
    assert (result.window_start_ms, result.window_end_ms, result.sample_count) == (1000, 6000, 3)
    assert (result.avg_headcount, result.peak_headcount) == (4, 6)


def test_half_up_not_bankers_rounding():
    agg = WindowAggregator()
    agg.add_sample(2, 1000)
    agg.add_sample(3, 2000)
    assert agg.flush(6000).avg_headcount == 3


def test_empty_window_not_emitted_and_average_never_exceeds_peak():
    agg = WindowAggregator()
    assert agg.flush(5000) is None
    agg.add_sample(0, 6000)
    summary = agg.flush(11000)
    assert summary.avg_headcount <= summary.peak_headcount
