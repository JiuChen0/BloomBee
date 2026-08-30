from pathlib import Path


def test_speculative_fallback_does_not_synthesize_eos_on_error():
    source = Path("src/bloombee/models/llama/speculative_model.py").read_text()
    method_start = source.index("    def _fallback_generation_with_forward(")
    method_end = source.index("    def _build_speculative_trees_batched(", method_start)
    method_source = source[method_start:method_end]

    except_start = method_source.index("        except Exception as e:")
    except_source = method_source[except_start:]

    assert "past_key_values.set_is_spec_decoding(old_spec_flag)" in except_source
    assert "logger.exception(" in except_source
    assert "raise" in except_source
    assert "torch.full((batch_size, 1), eos_token_id" not in except_source
    assert "empty_hidden" not in except_source

