#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The config-vs-checkpoint authority rule for policy processor pipelines.

`make_pre_post_processors` always builds structure from the policy config and takes only tensor state
from a checkpoint. These tests pin both halves of that rule, plus the one value deliberately carried
over from the checkpoint (the rename map).
"""

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import (
    _resolve_pretrained_processor_loader,
    make_policy_config,
    make_pre_post_processors,
)
from lerobot.processor import NormalizerProcessorStep, ProcessorBuildContext, UnnormalizerProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

ARTIFACTS_DIR = Path(__file__).parents[1] / "artifacts" / "processors"
sys.path.insert(0, str(ARTIFACTS_DIR))
from generate_golden_processor_configs import (  # noqa: E402
    _build_configs,
    _is_missing_optional_dependency,
)

PRE_FILENAME = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
POST_FILENAME = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"

STATE_DIM = 6
ACTION_DIM = 6


def _act_config(**overrides):
    config = make_policy_config("act", push_to_hub=False)
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        f"{OBS_IMAGE}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    config.device = "cpu"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _stats(fill: float) -> dict[str, dict[str, torch.Tensor]]:
    """Stats whose values identify which source they came from."""

    def block(shape):
        return {
            "mean": torch.full(shape, fill),
            "std": torch.ones(shape),
            "min": torch.full(shape, -fill),
            "max": torch.full(shape, fill),
        }

    return {
        OBS_STATE: block((STATE_DIM,)),
        ACTION: block((ACTION_DIM,)),
        f"{OBS_IMAGE}.cam": block((3, 1, 1)),
    }


def _save_checkpoint(tmp_path: Path, config, stats) -> None:
    preprocessor, postprocessor = make_pre_post_processors(
        config, context=ProcessorBuildContext(dataset_stats=stats)
    )
    preprocessor.save_pretrained(tmp_path, config_filename=PRE_FILENAME)
    postprocessor.save_pretrained(tmp_path, config_filename=POST_FILENAME)


def _normalizer(pipeline):
    return next(step for step in pipeline.steps if isinstance(step, NormalizerProcessorStep))


def _unnormalizer(pipeline):
    return next(step for step in pipeline.steps if isinstance(step, UnnormalizerProcessorStep))


def test_config_beats_checkpoint_for_structure(tmp_path):
    """A config value changed since the checkpoint was written takes effect on load.

    Under the old deserializing loader the saved pipeline was authoritative, so a `--policy.*` flag
    that should reconfigure a step silently did nothing.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a second real device to tell the config's choice from the checkpoint's")

    _save_checkpoint(tmp_path, _act_config(device="cpu"), _stats(1.0))

    reloaded, _ = make_pre_post_processors(
        _act_config(device="cuda"), pretrained_path=str(tmp_path), context=ProcessorBuildContext()
    )

    device_steps = [step for step in reloaded.steps if getattr(step, "device", None) is not None]
    assert device_steps, "expected the preprocessor to contain a device step"
    assert all("cuda" in str(step.device) for step in device_steps)


def test_finetune_stats_from_dataset_beat_checkpoint(tmp_path):
    """Supplying dataset stats makes the dataset authoritative — the finetune case."""
    _save_checkpoint(tmp_path, _act_config(), _stats(1.0))

    reloaded, _ = make_pre_post_processors(
        _act_config(),
        pretrained_path=str(tmp_path),
        context=ProcessorBuildContext(dataset_stats=_stats(9.0)),
    )

    assert torch.allclose(
        _normalizer(reloaded)._tensor_stats[OBS_STATE]["mean"], torch.full((STATE_DIM,), 9.0)
    )


def test_resume_stats_come_from_checkpoint(tmp_path):
    """Omitting dataset stats keeps the checkpoint's — the eval and resume case."""
    _save_checkpoint(tmp_path, _act_config(), _stats(1.0))

    pre, post = make_pre_post_processors(
        _act_config(), pretrained_path=str(tmp_path), context=ProcessorBuildContext()
    )

    assert torch.allclose(_normalizer(pre)._tensor_stats[OBS_STATE]["mean"], torch.full((STATE_DIM,), 1.0))
    assert torch.allclose(_unnormalizer(post)._tensor_stats[ACTION]["mean"], torch.full((ACTION_DIM,), 1.0))


def test_rename_map_is_carried_over_from_the_checkpoint(tmp_path):
    """A rename map exists only on the CLI, so rebuilding must not drop the checkpoint's."""
    _save_checkpoint(tmp_path, _act_config(rename_map={"left": "cam"}), _stats(1.0))

    config = _act_config()
    assert config.rename_map == {}
    reloaded, _ = make_pre_post_processors(
        config, pretrained_path=str(tmp_path), context=ProcessorBuildContext()
    )

    rename_step = reloaded.steps[0]
    assert rename_step.rename_map == {"left": "cam"}


def test_rename_map_from_the_config_wins(tmp_path):
    """An explicitly supplied rename map is not overwritten by the checkpoint's."""
    _save_checkpoint(tmp_path, _act_config(rename_map={"left": "cam"}), _stats(1.0))

    reloaded, _ = make_pre_post_processors(
        _act_config(rename_map={"right": "cam"}),
        pretrained_path=str(tmp_path),
        context=ProcessorBuildContext(),
    )

    assert reloaded.steps[0].rename_map == {"right": "cam"}


def test_legacy_step_overrides_still_apply_with_a_deprecation_warning(tmp_path):
    """Out-of-tree callers passing the old override dicts keep working."""
    _save_checkpoint(tmp_path, _act_config(), _stats(1.0))

    with pytest.warns(DeprecationWarning, match="Step-level processor overrides"):
        reloaded, _ = make_pre_post_processors(
            _act_config(),
            pretrained_path=str(tmp_path),
            preprocessor_overrides={
                "rename_observations_processor": {"rename_map": {"left": "cam"}},
                # Shape is owned by the policy factory now; this key must be ignored, not fatal.
                "normalizer_processor": {"features": {}},
            },
        )

    assert reloaded.steps[0].rename_map == {"left": "cam"}
    # The ignored key must not have wiped the features the factory computed.
    assert _normalizer(reloaded).features


def test_checkpoint_state_for_an_unbuilt_step_is_not_silently_dropped(tmp_path):
    """The one failure mode that would otherwise surface as bad numbers rather than an error."""
    _save_checkpoint(tmp_path, _act_config(), _stats(1.0))

    # Drop the action feature so the config builds an unnormalizer with no ACTION stats to load,
    # then hand the postprocessor's checkpoint to a pipeline that has no unnormalizer at all.
    pre_config = json.loads((tmp_path / PRE_FILENAME).read_text())
    pre_config["steps"].append(
        {"registry_name": "unnormalizer_processor", "config": {}, "state_file": "nonexistent.safetensors"}
    )
    (tmp_path / PRE_FILENAME).write_text(json.dumps(pre_config))

    with pytest.raises(ValueError, match="silently discarded"):
        make_pre_post_processors(
            _act_config(), pretrained_path=str(tmp_path), context=ProcessorBuildContext()
        )


def test_state_loads_when_the_checkpoints_step_indices_have_shifted(tmp_path):
    """State filenames embed the step index, so removing a step renames every later state file.

    Real case: pi0 pipelines used to carry a disabled `relative_actions_processor` ahead of the
    normalizer, putting its stats in `..._step_6_normalizer_processor.safetensors`. Rebuilding from
    the config now produces that normalizer at index 5. Pairing saved entries to built steps by step
    key rather than by position is what keeps those stats landing on the right step.
    """
    _save_checkpoint(tmp_path, _act_config(), _stats(1.0))

    # Simulate the checkpoint having been written with an extra stateless step up front.
    config_path = tmp_path / PRE_FILENAME
    saved = json.loads(config_path.read_text())
    normalizer_entry = next(s for s in saved["steps"] if s["registry_name"] == "normalizer_processor")
    old_state_file = normalizer_entry["state_file"]
    shifted_state_file = old_state_file.replace("_step_3_", "_step_9_")
    (tmp_path / shifted_state_file).write_bytes((tmp_path / old_state_file).read_bytes())
    normalizer_entry["state_file"] = shifted_state_file
    saved["steps"].insert(0, {"registry_name": "identity_processor", "config": {}})
    config_path.write_text(json.dumps(saved))

    pre, _post = make_pre_post_processors(
        _act_config(), pretrained_path=str(tmp_path), context=ProcessorBuildContext()
    )

    # The stats still land, despite the index in the filename not matching the rebuilt position.
    assert torch.allclose(_normalizer(pre)._tensor_stats[OBS_STATE]["mean"], torch.full((STATE_DIM,), 1.0))


@pytest.mark.parametrize("policy_type", sorted(PreTrainedConfig.get_known_choices()))
def test_every_policy_builds_from_config_alone(policy_type):
    """Every policy must build on the path eval and inference now use: config + bare context.

    This is the de-risking test for the policies the refactor does not touch.
    """
    try:
        _build_configs(policy_type)
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        if _is_missing_optional_dependency(reason):
            pytest.skip(f"{policy_type} is not installed in this environment")
        raise


@pytest.mark.parametrize(
    "policy_type", sorted(json.loads((ARTIFACTS_DIR / "golden" / "manifest.json").read_text())["covered"])
)
def test_serialized_pipeline_matches_the_golden_fixture(policy_type):
    """Rebuilding from the config must reproduce the structure captured before the refactor."""
    try:
        built = _build_configs(policy_type)
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        if _is_missing_optional_dependency(reason):
            pytest.skip(f"{policy_type} is not installed in this environment")
        raise

    for role, config in built.items():
        expected = json.loads((ARTIFACTS_DIR / "golden" / f"{policy_type}_{role}.json").read_text())
        assert config == expected, f"{policy_type} {role} pipeline drifted from its golden fixture"


# --- the documented opt-out: a policy that owns its own checkpoint loading --------------------------


def _register_fake_policy(monkeypatch, loader, policy_type="fakepolicy"):
    """A synthetic plugin policy whose `processor_*` module may define the pretrained-loader hook.

    Mirrors how the factory resolves real policies — sibling `configuration_*`/`processor_*` modules —
    without touching the policy registry, so nothing leaks into other tests.
    """
    module_root = f"lerobot_plugin_{policy_type}"
    processor_module = types.ModuleType(f"{module_root}.processor_{policy_type}")
    if loader is not None:
        setattr(processor_module, f"make_{policy_type}_pre_post_processors_from_pretrained", loader)
    monkeypatch.setitem(sys.modules, processor_module.__name__, processor_module)

    config_class = type(
        "FakePolicyConfig",
        (),
        {"__module__": f"{module_root}.configuration_{policy_type}", "type": policy_type},
    )
    return config_class()


def test_a_policy_can_own_its_checkpoint_loading_without_touching_core(monkeypatch, tmp_path):
    """Defining `make_{type}_pre_post_processors_from_pretrained` is the whole opt-in.

    The factory names no policy: the hook is resolved by the same naming convention as the regular
    processor factory, so a plugin can use it and migrating GR00T off it needs no core edit.
    """
    calls = []

    def loader(config, pretrained_path, *, revision=None, context=None, **_):
        calls.append({"path": pretrained_path, "revision": revision, "training": context.training})
        return "PRE", "POST"

    config = _register_fake_policy(monkeypatch, loader)

    result = make_pre_post_processors(
        config,
        pretrained_path=str(tmp_path),
        pretrained_revision="v1",
        context=ProcessorBuildContext(training=True),
    )

    assert result == ("PRE", "POST"), "the policy's loader owns the result; core must not rebuild"
    assert calls == [{"path": str(tmp_path), "revision": "v1", "training": True}]


def test_the_hook_is_ignored_when_no_checkpoint_is_given(monkeypatch):
    """It is a *pretrained* loader: building from scratch always goes through the normal factory."""

    def loader(config, pretrained_path, **_):
        raise AssertionError("the pretrained loader must not run without a checkpoint")

    config = _register_fake_policy(monkeypatch, loader)

    with pytest.raises(ValueError, match="not implemented"):
        make_pre_post_processors(config, context=ProcessorBuildContext())


def test_a_loader_receives_only_the_parameters_it_declares(monkeypatch, tmp_path):
    """A minimal `(config, pretrained_path)` signature is enough; the rest is opt-in by name."""

    def loader(config, pretrained_path):
        return f"PRE:{pretrained_path}", "POST"

    config = _register_fake_policy(monkeypatch, loader)

    pre, post = make_pre_post_processors(
        config, pretrained_path=str(tmp_path), context=ProcessorBuildContext()
    )

    assert (pre, post) == (f"PRE:{tmp_path}", "POST")


def test_a_loader_requiring_an_unsupplyable_parameter_is_rejected(monkeypatch, tmp_path):
    """Fail with the contract spelled out, rather than an opaque TypeError from the call."""

    def loader(config, pretrained_path, robot_kinematics):
        raise AssertionError("should not be reached")

    config = _register_fake_policy(monkeypatch, loader)

    with pytest.raises(ValueError, match="robot_kinematics"):
        make_pre_post_processors(config, pretrained_path=str(tmp_path), context=ProcessorBuildContext())


def test_a_policy_without_the_hook_rebuilds_from_its_config(monkeypatch, tmp_path):
    """The default for every policy, including one whose processor module exists but has no hook."""
    _save_checkpoint(tmp_path, _act_config(), _stats(1.0))

    assert _resolve_pretrained_processor_loader(_act_config()) is None

    pre, _ = make_pre_post_processors(
        _act_config(), pretrained_path=str(tmp_path), context=ProcessorBuildContext()
    )
    # Rebuilt from the config, then filled from the checkpoint.
    assert torch.allclose(_normalizer(pre)._tensor_stats[OBS_STATE]["mean"], torch.full((STATE_DIM,), 1.0))


def test_groot_is_the_only_in_tree_policy_using_the_hook():
    """Pins the current routing: GR00T opts out, everyone else rebuilds from config.

    Resolution only — instantiating GR00T's pipelines needs a gated backbone. When GR00T is migrated,
    deleting its loader is enough to flip it onto the rebuild path, and this test says so.
    """
    using_hook = []
    for policy_type in sorted(PreTrainedConfig.get_known_choices()):
        config_class = PreTrainedConfig.get_choice_class(policy_type)
        try:
            loader = _resolve_pretrained_processor_loader(config_class.__new__(config_class))
        except Exception as exc:  # noqa: BLE001
            if _is_missing_optional_dependency(f"{type(exc).__name__}: {exc}"):
                continue
            raise
        if loader is not None:
            using_hook.append(policy_type)

    assert using_hook in ([], ["groot"]), f"unexpected policies own their checkpoint loading: {using_hook}"
