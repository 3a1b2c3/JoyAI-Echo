"""Autoregressive text/image-to-video rollout for Echo-WM Flash."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_causal import (
    DEFAULT_CAUSAL_TIMESTEPS,
    CausalCacheConfig,
    CausalModelWrapper,
    causal_audio_frames,
    causal_rollout,
    causal_video_blocks,
)
from ltx_pipelines.utils import ModelLedger, assert_resolution, cleanup_memory, combined_image_conditionings, encode_prompts, get_device
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.helpers import create_noised_state, noise_video_state
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()

# Called after each streamed block is decoded, with newly-available pixel
# media only (not the running total). Args: (block_index, total_blocks,
# video_chunk_uint8 [f, h, w, c], audio_chunk_or_None). Opting in (passing
# on_block to __call__) keeps the video/audio decoders resident on GPU for
# the whole rollout instead of only after the transformer is freed, so it
# costs extra peak VRAM versus the non-streaming path.
OnBlockMedia = Callable[[int, int, torch.Tensor, Audio | None], None]


class CausalTI2VidPipeline:
    """Inference-only 4-step autoregressive I2V pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
        action_config=None,
        cache_config: CausalCacheConfig = CausalCacheConfig(),
    ) -> None:
        self.dtype = torch.bfloat16
        self.device = device
        self.action_config = action_config
        cache_config.validate()
        self.cache_config = cache_config
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )
        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)

        # Opt-in (ECHO_WM_COMPILE=1, default off): ModelLedger.transformer()
        # builds a brand-new model object from scratch on every __call__, so
        # torch.compile-ing it naively would mean paying full graph-tracing
        # cost on every single generation instead of once -- likely a net
        # loss, the same failure mode as the CUDA-cache-size tuning that
        # turned out not to help (see TROUBLESHOOTING.md). Caching the built
        # (and, if enabled, compiled) model here per (width, height) --
        # patches_per_frame and the action module are both derived from
        # those -- is what makes compiling actually pay off: it's built (and
        # compiled) once per resolution, then reused across every later
        # generation at that resolution instead of rebuilt from scratch.
        self._model_cache: dict[tuple[int, int, int, int], tuple] = {}
        self._compile_enabled = os.environ.get("ECHO_WM_COMPILE", "0") == "1"

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        action_cond: dict[str, torch.Tensor],
        timesteps: tuple[int, ...] | list[int] = DEFAULT_CAUSAL_TIMESTEPS,
        video_tiling_config: TilingConfig | None = None,
        on_block: OnBlockMedia | None = None,
        attn_window: tuple[int, int] | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        # attn_window overrides self.cache_config's (video_local_attn_size,
        # video_sink_size) for this call only, exposed so the UI can A/B
        # speed/coherence trade-offs live instead of only via config file +
        # restart. video_chunk_size is NOT overridable -- it must equal
        # CAUSAL_VIDEO_CHUNK_SIZE (enforced by CausalCacheConfig.validate()).
        if attn_window is not None:
            cache_config = CausalCacheConfig(
                video_local_attn_size=attn_window[0],
                video_sink_size=attn_window[1],
                video_chunk_size=self.cache_config.video_chunk_size,
            )
            cache_config.validate()
        else:
            cache_config = self.cache_config

        assert_resolution(height=height, width=width, is_two_stage=False)
        latent_frames = (num_frames - 1) // 8 + 1
        if num_frames != (latent_frames - 1) * 8 + 1:
            raise ValueError("causal --num-frames must be 1 + 8*n output frames")
        causal_video_blocks(latent_frames, cache_config.video_chunk_size)
        # The causal student is trained on one positive conditioning branch.
        encoded_prompt, = encode_prompts([prompt], self.model_ledger)
        if encoded_prompt.audio_encoding is None:
            raise ValueError("the causal AV checkpoint must provide audio text embeddings")

        output_shape = VideoPixelShape(1, num_frames, height, width, frame_rate)
        video_encoder = self.model_ledger.video_encoder()
        conditionings = combined_image_conditionings(
            images, height, width, video_encoder, self.dtype, self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        del video_encoder
        cleanup_memory()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator)
        video_state, video_tools = noise_video_state(
            output_shape, noiser, conditionings, self.pipeline_components,
            self.dtype, self.device,
        )
        audio_frames = causal_audio_frames(latent_frames, cache_config.video_chunk_size)
        audio_shape = AudioLatentShape(batch=1, channels=8, frames=audio_frames, mel_bins=16)
        audio_tools = AudioLatentTools(self.pipeline_components.audio_patchifier, audio_shape)
        audio_state = create_noised_state(
            audio_tools, [], noiser, self.dtype, self.device
        )

        # Keyed on attn_window too (not just resolution) -- different
        # windows need different KV-cache tensor sizes (see
        # CausalModelWrapper.init_caches), so they can't share a cached
        # model instance.
        cache_key = (width, height, cache_config.video_local_attn_size, cache_config.video_sink_size)
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            x0_model, wrapper = cached
        else:
            x0_model = self.model_ledger.transformer(action_config=self.action_config)
            if self._compile_enabled:
                t_compile = time.time()
                print(
                    f"[compile] wrapping transformer in torch.compile() for {width}x{height} "
                    f"(first call at this resolution only -- reused for every later generation "
                    f"at the same resolution). Actual graph tracing happens lazily on the first "
                    f"real forward call, not here, so this print returning fast is expected.",
                    flush=True,
                )
                # Confirmed on real hardware (see TROUBLESHOOTING.md item -4):
                # without this, torch.compile hits a repeated recompilation
                # storm from `if self.idx >= int(self.num_layers * 0.7)` in
                # transformer.py -- self.idx is fixed per layer instance, but
                # dynamo doesn't reliably treat plain nn.Module integer
                # attributes as static without this flag, so it kept
                # re-tracing instead of settling on one compiled graph
                # (warmup got stuck >100s on a single ~2s block). This is
                # dynamo's own suggested fix, per its error output --
                # untested whether it actually resolves the storm.
                # `import torch._dynamo` here (instead of `from torch import
                # _dynamo`) would bind the name `torch` locally -- Python's
                # `import a.b` binds `a`, not `a.b` -- making the whole
                # function treat `torch` as a local variable and breaking
                # every other `torch.*` use above this point in the function
                # (confirmed on real hardware: `UnboundLocalError: cannot
                # access local variable 'torch'` at the earlier
                # `torch.cuda.synchronize()` call). Import only the
                # submodule name instead.
                from torch import _dynamo  # noqa: PLC0415
                _dynamo.config.allow_unspec_int_on_nn_module = True
                x0_model.velocity_model = torch.compile(x0_model.velocity_model)
                print(f"[compile] torch.compile() wrap done in {time.time() - t_compile:.1f}s", flush=True)
            wrapper = CausalModelWrapper(
                x0_model.velocity_model,
                patches_per_frame=(height // 32) * (width // 32),
                cache=cache_config,
            )
            self._model_cache[cache_key] = (x0_model, wrapper)

        # Streaming preview (opt-in): build the decoders now so each block can
        # be decoded as soon as it's denoised, instead of only after the
        # transformer is freed. This holds decoders + transformer on GPU at
        # the same time, so it costs extra peak VRAM versus the default path.
        preview_video_decoder = preview_audio_decoder = preview_vocoder = None
        raw_on_block = None
        if on_block is not None:
            preview_video_decoder = self.model_ledger.video_decoder()
            preview_audio_decoder = self.model_ledger.audio_decoder()
            preview_vocoder = self.model_ledger.vocoder()
            # Frames/samples already handed to on_block, so each call only
            # emits the newly-decoded tail. The causal decoder only looks
            # backward in time, so already-emitted frames are stable as later
            # (still-zero) blocks are appended past them.
            seen = {"video_frames": 0, "audio_samples": 0}

            def raw_on_block(
                block_index: int,
                total_blocks: int,
                video_block,
                video_output_so_far: torch.Tensor,
                audio_output_so_far: torch.Tensor,
            ) -> None:
                preview_video_state = video_tools.unpatchify(video_tools.clear_conditioning(
                    video_state.__class__(
                        latent=video_output_so_far,
                        denoise_mask=video_state.denoise_mask,
                        positions=video_state.positions,
                        clean_latent=video_state.clean_latent,
                        attention_mask=None,
                    )
                ))
                preview_audio_state = audio_tools.unpatchify(audio_tools.clear_conditioning(
                    audio_state.__class__(
                        latent=audio_output_so_far,
                        denoise_mask=audio_state.denoise_mask,
                        positions=audio_state.positions,
                        clean_latent=audio_state.clean_latent,
                        attention_mask=None,
                    )
                ))
                decoded_video_so_far = next(vae_decode_video(
                    preview_video_state.latent, preview_video_decoder, tiling_config=None, generator=generator,
                ))
                decoded_audio_so_far = vae_decode_audio(
                    preview_audio_state.latent, preview_audio_decoder, preview_vocoder
                )

                # video_output_so_far is a fixed-size buffer for the whole
                # clip (rollout.py writes each block into its own slice, it
                # never grows) -- everything past this block's end is still
                # zero/undenoised placeholder. The VAE decodes the full
                # buffer regardless (it doesn't truncate itself just because
                # the tail is zero), so decoded_video_so_far's length never
                # actually grows between calls the way the old diffing logic
                # here assumed. Truncate to the pixel range this block
                # actually covers before diffing, using the same
                # latent<->pixel frame conversion the pipeline itself uses
                # (num_frames = (latent_frames - 1) * 8 + 1).
                _, video_end = video_block
                pixel_end = (video_end - 1) * 8 + 1
                decoded_so_far_valid = decoded_video_so_far[:pixel_end]

                # .clone() strips the inference-tensor flag: these chunks get
                # handed to a callback that runs later / on another thread
                # (outside this function's inference_mode/no_grad scope), and
                # inference tensors cannot be touched once that scope exits.
                video_chunk = decoded_so_far_valid[seen["video_frames"]:].clone()
                seen["video_frames"] = decoded_so_far_valid.shape[0]

                # Same fixed-size-buffer issue as video: audio_output_so_far
                # never grows either, so decoded_audio_so_far's waveform is
                # always the full-clip length. Truncate proportionally using
                # this block's target audio-latent-frame count vs. the total
                # (audio_frames, the full-clip audio latent length computed
                # at call start) against the full decoded waveform length --
                # avoids needing decoded_audio_so_far's exact sample-rate/
                # hop-size relationship to audio latent frames.
                target_audio_frames = causal_audio_frames(video_end, cache_config.video_chunk_size)
                waveform = decoded_audio_so_far.waveform
                sample_end = int(waveform.shape[-1] * target_audio_frames / audio_frames)
                waveform_valid = waveform[..., :sample_end]

                audio_chunk = None
                if waveform_valid.shape[-1] > seen["audio_samples"]:
                    audio_chunk = Audio(
                        waveform=waveform_valid[..., seen["audio_samples"]:].clone(),
                        sampling_rate=decoded_audio_so_far.sampling_rate,
                    )
                    seen["audio_samples"] = waveform_valid.shape[-1]

                on_block(block_index, total_blocks, video_chunk, audio_chunk)

        generated_video, generated_audio = causal_rollout(
            wrapper=wrapper,
            clean_video=video_state.clean_latent,
            clean_audio=audio_state.clean_latent,
            video_positions=video_state.positions,
            audio_positions=audio_state.positions,
            video_context=encoded_prompt.video_encoding,
            audio_context=encoded_prompt.audio_encoding,
            context_mask=encoded_prompt.attention_mask,
            action_cond=action_cond,
            seed=seed,
            timesteps=timesteps,
            on_block=raw_on_block,
        )
        del wrapper, x0_model
        cleanup_memory()

        video_state = video_tools.unpatchify(video_tools.clear_conditioning(
            video_state.__class__(
                latent=generated_video,
                denoise_mask=video_state.denoise_mask,
                positions=video_state.positions,
                clean_latent=video_state.clean_latent,
                attention_mask=None,
            )
        ))
        audio_state = audio_tools.unpatchify(audio_tools.clear_conditioning(
            audio_state.__class__(
                latent=generated_audio,
                denoise_mask=audio_state.denoise_mask,
                positions=audio_state.positions,
                clean_latent=audio_state.clean_latent,
                attention_mask=None,
            )
        ))
        decoded_video = vae_decode_video(
            video_state.latent,
            preview_video_decoder or self.model_ledger.video_decoder(),
            tiling_config=video_tiling_config,
            generator=generator,
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent,
            preview_audio_decoder or self.model_ledger.audio_decoder(),
            preview_vocoder or self.model_ledger.vocoder(),
        )
        return decoded_video, decoded_audio
