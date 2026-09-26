"""Song-level splits never leak a song_id across train, val, and test."""

import pytest

from hum2song.data.splits import (
    SplitLeakageError,
    SplitRow,
    assert_no_song_leakage,
    assign_song_splits,
    hash_split,
    propose_split,
)


def test_hash_split_is_deterministic() -> None:
    assert hash_split("humtrans:0042") == hash_split("humtrans:0042")


def test_holdout_groups_are_test_even_with_an_official_train_label() -> None:
    row = SplitRow("mirqbsh:00014", "mirqbsh", "train", "Twinkle")
    assert propose_split(row) == "test"
    assert propose_split(SplitRow("mlend:Potter", "mlend", None, "Potter")) == "test"
    assert propose_split(SplitRow("mtgqbh:girl", "mtgqbh", None, "Girl")) == "test"


def test_official_humtrans_split_is_kept() -> None:
    rows = [
        SplitRow("humtrans:0001", "humtrans", "train", None),
        SplitRow("humtrans:0002", "humtrans", "val", None),
        SplitRow("humtrans:0003", "humtrans", "test", None),
    ]
    assigned = assign_song_splits(rows)
    assert assigned == {"humtrans:0001": "train", "humtrans:0002": "val", "humtrans:0003": "test"}


def test_conflicting_official_labels_use_the_strictest_split() -> None:
    rows = [
        SplitRow("humtrans:0001", "humtrans", "train", None),
        SplitRow("humtrans:0001", "humtrans", "test", None),
    ]
    assert assign_song_splits(rows)["humtrans:0001"] == "test"


def test_title_overlap_moves_a_train_song_onto_the_eval_song() -> None:
    rows = [
        SplitRow("mirqbsh:00014", "mirqbsh", None, "Twinkle, Twinkle, Little Star"),
        SplitRow("humtrans:0009", "humtrans", "train", "twinkle twinkle little star"),
    ]
    assigned = assign_song_splits(rows)
    assert assigned["mirqbsh:00014"] == "test"
    assert assigned["humtrans:0009"] == "test"


def test_mirqbsh_singing_trains_without_leaking_the_held_out_song() -> None:
    rows = [
        SplitRow("mirqbsh:00001", "mirqbsh", None, "I'm the teapot", "sing"),
        SplitRow("mirqbsh:00001", "mirqbsh", None, "I'm the teapot", "hum"),
        SplitRow("mirqbsh:00014", "mirqbsh", None, "Twinkle", "hum"),
    ]
    assigned = assign_song_splits(rows)
    assert assigned["mirqbsh:00001"] == hash_split("mirqbsh:00001")
    assert assigned["mirqbsh:00001"] == "train"
    assert assigned["mirqbsh:00014"] == "test"


def test_leakage_check_rejects_a_song_in_two_splits() -> None:
    with pytest.raises(SplitLeakageError):
        assert_no_song_leakage([("song", "train"), ("song", "test")])


def test_leakage_check_accepts_one_split_per_song() -> None:
    assert_no_song_leakage([("a", "train"), ("a", "train"), ("b", "test")])
