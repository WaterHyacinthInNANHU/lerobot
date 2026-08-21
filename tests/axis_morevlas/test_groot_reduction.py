from __future__ import annotations

import torch


class _FakeGrootModel:
    """Returns a BatchFeature-like dict with the fields groot_n1_7's action head produces --
    including action_mask, which the real action head returns alongside action_loss (the mask
    it actually computed action_loss against)."""

    def __init__(self, action_loss, mask):
        self._out = {
            "loss": action_loss.sum() / (mask.sum() + 1e-6),
            "action_loss": action_loss,
            "action_mask": mask,
        }

    def forward(self, inputs):
        return self._out


def _policy_with_fake(action_loss, mask, input_mask=None):
    """`input_mask` lets a test make groot_inputs['action_mask'] disagree with the model
    output's action_mask, to pin that reduction='none' sources the mask from the output, not
    the input. Defaults to `mask` so callers that don't care see the two agree, as before."""
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    p = GrootPolicy.__new__(GrootPolicy)  # skip heavy __init__: unit-test the forward math only
    torch.nn.Module.__init__(p)  # minimal nn.Module state so get_device_from_parameters(self) works
    p.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))
    p._groot_model = _FakeGrootModel(action_loss, mask)
    p.config = type("C", (), {"use_bf16": False})()
    resolved_input_mask = mask if input_mask is None else input_mask
    p._filter_groot_inputs = lambda batch, include_action=True: {"action_mask": resolved_input_mask}
    return p


def test_reduction_none_matches_scalar_recombination():
    torch.manual_seed(0)
    b, h, d = 4, 16, 7
    mask = torch.ones(b, h, d)
    mask[2, 10:] = 0  # one short episode tail
    raw = torch.rand(b, h, d) * mask
    p = _policy_with_fake(raw, mask)

    scalar, _ = p.forward({"action_mask": mask})
    per_sample, _ = p.forward({"action_mask": mask}, reduction="none")

    assert per_sample.shape == (b,)
    mask_sums = mask.sum(dim=(1, 2))
    recombined = (per_sample * (mask_sums + 1e-6)).sum() / (mask_sums.sum() + 1e-6)
    torch.testing.assert_close(recombined, scalar, rtol=1e-5, atol=1e-6)


def test_mean_path_unchanged_signature():
    mask = torch.ones(2, 4, 7)
    p = _policy_with_fake(torch.rand(2, 4, 7), mask)
    loss, d = p.forward({"action_mask": mask})
    assert loss.ndim == 0 and "loss" in d


def test_reduction_none_uses_model_output_mask_not_input_mask():
    """Regression pin: the per-sample mask must come from outputs['action_mask'] (what
    action_loss was actually computed against), not groot_inputs['action_mask'] -- construct
    them to disagree so a fallback-to-input regression would be caught."""
    torch.manual_seed(0)
    b, h, d = 4, 16, 7
    output_mask = torch.ones(b, h, d)
    output_mask[2, 10:] = 0  # one short episode tail, visible only on the output mask
    input_mask = torch.ones(b, h, d)  # deliberately disagrees: claims a full mask
    raw = torch.rand(b, h, d) * output_mask
    p = _policy_with_fake(raw, output_mask, input_mask=input_mask)

    per_sample, _ = p.forward({"action_mask": input_mask}, reduction="none")

    expected = raw.sum(dim=(1, 2)) / (output_mask.sum(dim=(1, 2)) + 1e-6)
    torch.testing.assert_close(per_sample, expected, rtol=1e-5, atol=1e-6)
