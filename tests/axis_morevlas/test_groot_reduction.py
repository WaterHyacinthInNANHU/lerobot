from __future__ import annotations

import torch


class _FakeGrootModel:
    """Returns a BatchFeature-like dict with the fields groot_n1_7's action head produces."""

    def __init__(self, action_loss, mask):
        self._out = {"loss": action_loss.sum() / (mask.sum() + 1e-6),
                     "action_loss": action_loss}

    def forward(self, inputs):
        return self._out


def _policy_with_fake(action_loss, mask):
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    p = GrootPolicy.__new__(GrootPolicy)  # skip heavy __init__: unit-test the forward math only
    torch.nn.Module.__init__(p)  # minimal nn.Module state so get_device_from_parameters(self) works
    p.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))
    p._groot_model = _FakeGrootModel(action_loss, mask)
    p.config = type("C", (), {"use_bf16": False})()
    p._filter_groot_inputs = lambda batch, include_action=True: {"action_mask": mask}
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
