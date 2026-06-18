from avion.llm.rs_flag import evaluate_rs_flag


def test_rs_flag_positive_caption():
    result = evaluate_rs_flag("An aerial view of an airport with runways and terminal aprons.")
    assert result.rs_flag == 1
    assert "aerial view" in result.positive_terms_detected


def test_rs_flag_negative_caption():
    result = evaluate_rs_flag("A close-up street view of an indoor airport terminal.")
    assert result.rs_flag == 0
    assert result.negative_terms_detected


def test_rs_flag_length_constraint():
    result = evaluate_rs_flag("Aerial view.")
    assert result.rs_flag == 0
    assert "too_short" in result.reasons

