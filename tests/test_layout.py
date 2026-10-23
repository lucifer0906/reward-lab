import pytest

from reward_rl.envs.layout import LayoutError, parse_layout


def test_parse_base_layout(base_layout):
    L = base_layout
    assert (L.height, L.width) == (5, 17)
    assert L.num_pairs == 3
    assert L.start == (1, 1)
    assert L.goal == (1, 15)
    assert L.key_positions == ((3, 1), (2, 6), (3, 10))
    assert L.door_positions == ((2, 4), (2, 8), (2, 12))


@pytest.mark.parametrize(
    "rows, msg",
    [
        ([], "empty"),
        (["#####", "#SaAG#"], "equal length"),
        (["#aAG#"], "exactly one 'S'"),
        (["#SaA#"], "exactly one 'G'"),
        (["#SaG#"], "pair up"),
        (["#SbBG#"], "contiguous"),
        (["#SaAGx#"], "Unknown character"),
        (["#SaaAG#"], "Duplicate key"),
    ],
)
def test_invalid_layouts_rejected(rows, msg):
    with pytest.raises(LayoutError, match=msg):
        parse_layout(rows)
