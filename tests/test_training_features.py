from poker44.training.features import FEATURE_VERSION, N_FEATURES, featurize_chunk


def test_featurize_shape_and_version_tag():
    hand = {
        "metadata": {"bb": 0.05},
        "players": [{"seat": 1}],
        "streets": [{"street": "flop", "board_cards": []}],
        "actions": [
            {
                "action_type": "call",
                "street": "preflop",
                "normalized_amount_bb": 1.0,
                "pot_after": 1.0,
            }
        ],
        "outcome": {},
    }
    v = featurize_chunk([hand])
    assert v.shape == (N_FEATURES,)
    assert abs(float(v[33]) - FEATURE_VERSION / 10.0) < 1e-5
