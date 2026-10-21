from types import SimpleNamespace
import pytest
from unittest.mock import patch
from flux_mvwd.preparation import fit_caption


def test_caption_budget_preserves_exact_geometry_and_records_original():
    processor = SimpleNamespace(apply_chat_template=lambda messages, **k: messages[0]["content"],
                                tokenizer=lambda text: {"input_ids": list(text)})
    with patch("flux3.text_encoder.MAX_LENGTH", 160):
        record = fit_caption(processor, "a" * 200, "K=[100,0,32];T=identity")
        assert record["prompt"].endswith("K=[100,0,32];T=identity")
        assert record["raw_upsampler_caption"] == "a" * 200
        assert record["caption_shortened_to_preserve_geometry"]
        assert record["text_tokens"] <= 152
        unchanged = fit_caption(processor, "short", "geometry")
        assert not unchanged["caption_shortened_to_preserve_geometry"]
        with pytest.raises(ValueError, match="constraints alone"):
            fit_caption(processor, "caption", "g" * 200)
