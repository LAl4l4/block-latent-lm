from experiments.evaluate_pair_generation import comparison, summarize


def test_comparison_tracks_partial_and_missing_second_blocks():
    values = comparison([1, 2, 3, 4, 5], [1, 2, 9], 2, 6)
    assert values["first_block_match"]
    assert values["full"]["matched"] == 2
    assert values["full"]["compared"] == 3
    assert values["full"]["max_length"] == 5
    assert values["second_block"]["both_present"]
    assert values["second_block"]["whole_match"] is False
    assert values["blocks"][2]["whole_match"] is None
