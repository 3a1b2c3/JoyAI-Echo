import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

WM_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(WM_ROOT),
    str(WM_ROOT / "ltx-core" / "src"),
    str(WM_ROOT / "ltx-causal" / "src"),
]

from helpers.action_condition import build_causal_action_condition  # noqa: E402
from ltx_core.model.transformer.attention import update_kv_cache  # noqa: E402
from ltx_core.model.transformer.transformer import rebase_viewmat_translation  # noqa: E402
import ltx_causal as causal  # noqa: E402


def test_four_step_schedule_has_four_student_steps_and_no_zero():
    sigmas = causal.resolve_causal_sigmas()
    assert len(sigmas) == 4
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
    assert sigmas[-1] > 0


def test_default_241_frame_block_layout_and_audio_mapping():
    blocks = causal.causal_video_blocks(31)
    assert blocks == [(0, 1), *[(start, start + 3) for start in range(1, 31, 3)]]
    assert causal.causal_audio_frames(31) == 252
    assert causal.causal_audio_blocks(31)[-1] == (227, 252)
    cache = causal.CausalCacheConfig()
    assert (
        cache.video_local_attn_size,
        cache.video_sink_size,
        cache.video_chunk_size,
    ) == (19, 7, 3)
    assert (cache.audio_local_attn_size, cache.audio_sink_size) == (152, 52)


def test_audio_cache_sizes_follow_video_cache_alignment():
    cache = causal.CausalCacheConfig(
        video_local_attn_size=25,
        video_sink_size=7,
    )
    cache.validate()
    assert (cache.audio_local_attn_size, cache.audio_sink_size) == (202, 52)

    invalid = causal.CausalCacheConfig(video_local_attn_size=20)
    with pytest.raises(ValueError, match="audio alignment"):
        invalid.validate()


def test_flash_rejects_unsupported_video_chunk_size():
    cache = causal.CausalCacheConfig(video_chunk_size=4)
    with pytest.raises(ValueError, match="requires video_chunk_size=3"):
        cache.validate()
    with pytest.raises(ValueError, match="requires video_chunk_size=3"):
        causal.causal_video_blocks(31, chunk_size=0)
    with pytest.raises(ValueError, match="latent video length"):
        causal.causal_audio_frames(30)


def test_sink_plus_fifo_cache_rollover_and_block_replacement():
    cache = {
        "k": torch.zeros(1, 7, 1), "v": torch.zeros(1, 7, 1),
        "positions": torch.full((7,), -1, dtype=torch.long), "length": 0,
        "local_attn_size": 7, "sink_tokens": 2,
    }
    with torch.no_grad():
        for start in (0, 2, 5, 8):
            values = torch.arange(start, start + 3).view(1, 3, 1).float()
            update_kv_cache(cache, start, values, values)
    assert cache["positions"][: cache["length"]].tolist() == [0, 1, 6, 7, 8, 9, 10]
    replacement = torch.full((1, 3, 1), 99.0)
    with torch.no_grad():
        active_k, _ = update_kv_cache(cache, 8, replacement, replacement)
    assert active_k[0, -3:, 0].tolist() == [99.0, 99.0, 99.0]


def test_bounded_anchor_translation_preserves_relative_camera_transform():
    angle = torch.tensor(0.7)
    rotation = torch.tensor([
        [torch.cos(angle), 0.0, torch.sin(angle)],
        [0.0, 1.0, 0.0],
        [-torch.sin(angle), 0.0, torch.cos(angle)],
    ])
    cameras = torch.eye(4).repeat(1, 2, 1, 1)
    cameras[0, 0, :3, 3] = torch.tensor([3.0, 1.0, -2.0])
    cameras[0, 1, :3, :3] = rotation
    cameras[0, 1, :3, 3] = torch.tensor([-1.0, 4.0, 2.0])
    before = cameras[:, 0] @ torch.linalg.inv(cameras[:, 1])
    rebased = rebase_viewmat_translation(cameras, cameras[:, :1])
    after = rebased[:, 0] @ torch.linalg.inv(rebased[:, 1])
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rebased[:, 0, :3, 3], torch.zeros(1, 3), atol=1e-6, rtol=0)


def test_causal_action_path_keeps_fp32_cameras():
    condition = build_causal_action_condition(
        "wj-8", num_frames=9, width=64, height=64,
        translation_speed=0.025, rotation_speed_deg=0.6,
        pitch_limit_deg=60.0, fov_deg=70.0,
        device=torch.device("cpu"), fps=24.0,
    )
    assert all(value.dtype == torch.float32 for value in condition.values())


def test_causal_cli_has_no_cfg_and_accepts_both_cache_flag_spellings():
    source = (WM_ROOT / "inference_wm_causal.py").read_text(encoding="utf-8")
    assert "negative-prompt" not in source and "video-cfg" not in source and "audio-cfg" not in source
    for spelling in (
        "--video-local-attn-size", "--video_local_attn_size",
        "--video-sink-size", "--video_sink_size",
        "--video-chunk-size", "--video_chunk_size",
    ):
        assert spelling in source
    pipeline = (WM_ROOT / "ltx-pipelines" / "src" / "ltx_pipelines" / "causal_ti2vid.py").read_text()
    assert "encode_prompts([prompt]" in pipeline


def test_causal_case_runner_dry_run_uses_causal_entrypoint():
    result = subprocess.run(
        [sys.executable, str(WM_ROOT / "scripts" / "run_wm_case_causal.py"),
         "--case", "examples/wm_causal_cases/0104", "--dry-run"],
        cwd=WM_ROOT, check=True, capture_output=True, text=True,
    )
    assert "inference_wm_causal.py" in result.stdout
    assert "echo-wm-flash.safetensors" in result.stdout
    assert "--num-frames 385" in result.stdout
    assert "--fov-deg 70.0" in result.stdout
    assert "--translation-speed" not in result.stdout
    assert "--rotation-speed-deg" not in result.stdout
    assert "--pitch-limit-deg" not in result.stdout
    assert "--video-local-attn-size" not in result.stdout


def test_causal_multigpu_runner_uses_active_python_environment():
    source = (WM_ROOT / "scripts" / "run_wm_causal_cases_multigpu.sh").read_text()
    assert 'python_bin="${PYTHON_BIN:-python}"' in source


def test_checked_in_wbench_causal_cases_have_four_4_second_actions():
    expected_actions = {
        "0081": "k-96,i-96,s-96,w-96",
        "0104": "j-96,j-96,l-96,l-96",
        "0170": "w-96,s-96,a-96,l-96",
    }
    expected_prompts = {
        "0081": "A sunlit artist's studio with exposed brick walls and wooden shelves holding art supplies. A canvas sits on a wooden easel showing a half-finished landscape painting. Paint tubes, brushes, and palettes are scattered on a worktable. A tall arched window lets in bright natural light, with potted plants and ferns on the sill. Framed sketches hang on the walls. Above, exposed wooden ceiling beams support a hanging pendant lamp and a dried flower wreath. Below the worktable, paint-stained rags, a jar of turpentine, and stacked canvases lean against the wall. Behind the viewer, a cluttered bookshelf holds art reference books, a ceramic coffee mug, and a small plaster bust. First-person viewer. First-person view with the right hand holding a wooden paintbrush tipped with blue paint, extending toward the canvas.",
        "0104": "A floating sky island with lush green grass and colorful wildflowers on the clifftop. Waterfalls cascade off the island edges into clouds far below. Other floating islands visible in the distance. A crystal tree with glowing fruit stands on the right. Blue sky with fluffy clouds. Anime fantasy style with vibrant colors and ethereal lighting. Off-screen to the left: a stone fairy ring arch covered in glowing vines, leading to a hidden garden with luminous mushrooms. Off-screen to the right (behind the crystal tree): a cliff edge where a wooden rope bridge extends toward a neighboring floating island with a ruined tower. A fairy girl in a white dress with a silver tiara and translucent dragonfly wings that shimmer in the light. Blonde hair, seen from behind. Wings flutter gently, hair sways softly in the breeze. Third-person perspective, rear view, medium shot following the fairy girl who is at horizontal center of the frame.",
        "0170": "A grand mythological hall rendered in oil-painting style with visible brushstroke textures. Tall marble pillars with ornate Corinthian capitals line both sides of a wide corridor. Voluminous golden clouds billow between and beyond the pillars, filling the background with ethereal light. A large luminous archway glows at the far end of the hall. The floor is polished marble, and the overall palette is warm gold, cream, and blue, evoking a classical Renaissance or Baroque painting of Mount Olympus. Behind the viewer, additional marble pillars recede into a second chamber with a vaulted ceiling painted with celestial figures. To the left, the spaces between pillars open onto a cloud-filled void with distant mountain peaks below. To the right, a stone balustrade overlooks a vast golden cloudscape lit by an unseen divine light source. The luminous archway ahead leads deeper into the divine realm. A god-like male figure seen from behind, wearing flowing robes in deep blue and gold that trail behind him with heavy fabric dynamics. He has bare feet and walks forward with a purposeful stride. A faint halo or nimbus of light encircles his head. The brushstroke texture of the oil-painting style is visible on his robes and skin. Third-person rear-follow camera positioned directly behind the figure at mid-torso height. The figure is centered in the lower portion of the frame, walking forward through the hall of pillars toward the glowing archway. The camera tracks at a stable distance, maintaining the painterly composition.",
    }
    root = WM_ROOT / "examples" / "wm_causal_cases"
    for name, action in expected_actions.items():
        case = json.loads((root / name / "case.json").read_text(encoding="utf-8"))
        assert set(case) == {"prompt", "action", "fov_deg", "seed"}
        assert (root / name / "input.jpg").is_file()
        assert case["action"] == action
        assert case["seed"] == 42
        assert all(segment.endswith("-96") for segment in action.split(","))
        assert case["prompt"] == expected_prompts[name]


def test_causal_config_has_wbench_camera_and_bounded_cache_defaults():
    config = yaml.safe_load((WM_ROOT / "configs" / "inference_wm_causal.yaml").read_text())
    assert config["model"]["checkpoint"] == "checkpoints/echo-wm-flash.safetensors"
    assert config["action"] == {
        "enabled": True,
        "ucpe": True,
        "translation_speed": 0.05,
        "rotation_speed_deg": 0.4,
        "pitch_limit_deg": 40.0,
        "fov_deg": 70.0,
    }
    assert config["causal"] == {
        "timesteps": [1000, 750, 500, 250],
        "video_local_attn_size": 19,
        "video_sink_size": 7,
        "video_chunk_size": 3,
    }
