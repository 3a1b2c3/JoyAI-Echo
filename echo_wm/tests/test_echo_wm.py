import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
WM_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WM_ROOT), str(WM_ROOT / "ltx-core" / "src")]

from helpers.action_camera import (
    DEFAULT_ROTATION_SPEED_DEG,
    DEFAULT_TRANSLATION_SPEED,
    default_k_pix,
    parse_action_string,
)
from helpers.action_condition import action_config, build_action_condition
from helpers.moge_fov import effective_fov_x
from helpers.action_overlay import _normalised_rotation, _translation_keys
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig


def test_action_dsl_parses_combined_and_idle_segments():
    frames = parse_action_string("w-2, wj-1, none-2")
    assert frames == [["w"], ["w"], ["j", "w"], [], []]


@pytest.mark.parametrize("value", ["", "w", "x-3", "w-0", "w-nope"])
def test_action_dsl_rejects_invalid_input(value):
    with pytest.raises(ValueError):
        parse_action_string(value)


def test_action_condition_shape_dtype_and_keys():
    condition = build_action_condition(
        "w-4", num_frames=17, width=64, height=64,
        translation_speed=0.025, rotation_speed_deg=0.6,
        pitch_limit_deg=60.0, fov_deg=70.0, device=torch.device("cpu"), fps=24.0,
    )
    assert set(condition) == {"ucpe_viewmats", "ucpe_Ks"}
    assert condition["ucpe_viewmats"].shape == (1, 3, 4, 4)
    assert condition["ucpe_Ks"].shape == (1, 3, 3, 3)
    assert all(value.dtype == torch.bfloat16 for value in condition.values())


def test_ucpe_attention_shape_and_device():
    cfg = action_config(width=64, height=64, num_blocks=1)
    cfg.ucpe_attn_dim = 32
    cfg.ucpe_num_heads = 2
    block = BasicAVTransformerBlock(
        idx=0, num_layers=1,
        video=TransformerConfig(dim=64, heads=2, d_head=32, context_dim=64),
    )
    block._init_action_params(TransformerConfig(dim=64, heads=2, d_head=32, context_dim=0), cfg)
    condition = build_action_condition(
        "w-8", num_frames=9, width=64, height=64,
        translation_speed=0.025, rotation_speed_deg=0.6,
        pitch_limit_deg=60.0, fov_deg=70.0, device=torch.device("cpu"), fps=24.0,
    )
    x = torch.randn(1, 8, 64)
    out = block._apply_ucpe_attention(
        x, condition["ucpe_viewmats"].float(), condition["ucpe_Ks"].float()
    )
    assert out.shape == x.shape
    assert out.dtype == x.dtype and out.device == x.device


def test_ucpe_attention_accepts_fp32_cameras_with_bf16_hidden_states():
    cfg = action_config(width=64, height=64, num_blocks=1)
    cfg.ucpe_attn_dim = 32
    cfg.ucpe_num_heads = 2
    block = BasicAVTransformerBlock(
        idx=0, num_layers=1,
        video=TransformerConfig(dim=64, heads=2, d_head=32, context_dim=64),
    ).to(torch.bfloat16)
    block._init_action_params(TransformerConfig(dim=64, heads=2, d_head=32, context_dim=0), cfg)
    block = block.to(torch.bfloat16)
    block.ucpe_prope.coeffs_x_0 = block.ucpe_prope.coeffs_x_1 = None
    block.ucpe_prope.coeffs_y_0 = block.ucpe_prope.coeffs_y_1 = None
    condition = build_action_condition(
        "w-8", num_frames=9, width=64, height=64,
        translation_speed=0.025, rotation_speed_deg=0.6,
        pitch_limit_deg=60.0, fov_deg=70.0, device=torch.device("cpu"), fps=24.0,
    )
    x = torch.randn(1, 8, 64, dtype=torch.bfloat16)
    out = block._apply_ucpe_attention(
        x, condition["ucpe_viewmats"].float(), condition["ucpe_Ks"].float()
    )
    assert out.shape == x.shape and out.dtype == torch.bfloat16


def test_fov_crop_and_default_intrinsics():
    effective, factor = effective_fov_x(90.0, 200, 100, 100, 100)
    assert factor == pytest.approx(0.5)
    assert effective == pytest.approx(53.130102, rel=1e-6)
    K = default_k_pix(1280, 704, 70.0)
    assert K[0, 2].item() == 640 and K[1, 2].item() == 352



def test_prompt_skill_has_required_six_fields():
    skill = (WM_ROOT / "PROMPT_SKILL.md").read_text()
    for field in ("Environment:", "Character:", "Style:", "Perspective:", "Sounds:", "Speech:"):
        assert field in skill
    assert "Do not request subtitles" in skill


def test_public_config_has_only_semantic_action_controls():
    text = (WM_ROOT / "configs" / "inference_wm.yaml").read_text()
    lowered = text.lower()
    assert "translation_speed:" in text
    assert "rotation_speed_deg:" in text
    assert DEFAULT_TRANSLATION_SPEED > 0
    assert DEFAULT_ROTATION_SPEED_DEG > 0
    assert "global_trans" not in lowered and "normalize_mode" not in lowered
    assert "video_cfg: 4.0" in text
    assert "audio_cfg: 2.0" in text
    assert "negative_prompt:" in text
    assert "game UI" in text
    assert "crosshair" in text


def test_action_overlay_derives_stable_hud_controls():
    trans = __import__("numpy").array([[0.0, 0.0, 0.2], [0.1, 0.0, 0.0]], dtype=float)
    keys = _translation_keys(trans)
    yaw, pitch = _normalised_rotation(__import__("numpy").zeros((2, 3), dtype=float))
    assert len(keys) == 3 and keys[0] == ["W"]
    assert yaw.shape == pitch.shape == (3,)
