import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def test_deep_prompt_parameter_count():
    from avion.models.deep_prompt import DeepPromptSet

    prompts = DeepPromptSet(
        vision_layers=12,
        vision_tokens=8,
        vision_width=768,
        text_layers=12,
        text_tokens=4,
        text_width=512,
    )
    assert prompts.num_parameters() == 12 * 8 * 768 + 12 * 4 * 512

