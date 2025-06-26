"""Clean Kokoro implementation with controlled resource management."""

import os
from functools import lru_cache
from typing import AsyncGenerator, Dict, Optional, Tuple, Union

import numpy as np
import torch
from kokoro import KModel, KPipeline
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import model_config
from ..structures.schemas import WordTimestamp
from .base import AudioChunk, BaseModelBackend


class OptimizedKPipeline(KPipeline):
    def load_voice(self, voice: Union[str, torch.Tensor], delimiter: str = ",") -> torch.Tensor:
        # For gpu tensor, FloatTensor is not supported
        if isinstance(voice, torch.Tensor):
            return voice
        if voice in self.voices:
            return self.voices[voice]
        logger.debug(f"Loading voice: {voice}")
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        if len(packs) == 1:
            return packs[0]
        self.voices[voice] = torch.mean(torch.stack(packs), dim=0)
        return self.voices[voice]


class KokoroV1(BaseModelBackend):
    """Kokoro backend with controlled resource management."""

    def __init__(self):
        """Initialize backend with environment-based configuration."""
        super().__init__()
        # Strictly respect settings.use_gpu
        self._device = settings.get_device()
        self._model: Optional[KModel] = None
        self._pipelines: Dict[str, OptimizedKPipeline] = {}  # Store pipelines by lang_code
        # Cache for voice tensors - key: voice_name, value: tensor
        self._voice_cache: Dict[str, torch.Tensor] = {}
        self._voice_cache_size = 50  # Maximum number of cached voices

    async def load_model(self, path: str) -> None:
        """Load pre-baked model.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

            if not os.path.exists(config_path):
                raise RuntimeError(f"Config file not found: {config_path}")

            logger.info(f"Loading Kokoro model on {self._device}")
            logger.info(f"Config path: {config_path}")
            logger.info(f"Model path: {model_path}")

            # Load model and let KModel handle device mapping
            self._model = KModel(config=config_path, model=model_path).eval()
            # For MPS, manually move ISTFT layers to CPU while keeping rest on MPS
            if self._device == "mps":
                logger.info(
                    "Moving model to MPS device with CPU fallback for unsupported operations"
                )
                self._model = self._model.to(torch.device("mps"))
            elif self._device == "cuda":
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    def _get_pipeline(self, lang_code: str) -> OptimizedKPipeline:
        """Get or create pipeline for language code.

        Args:
            lang_code: Language code to use

        Returns:
            OptimizedKPipeline instance for the language
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        if lang_code not in self._pipelines:
            logger.info(f"Creating new pipeline for language code: {lang_code}")
            self._pipelines[lang_code] = OptimizedKPipeline(
                lang_code=lang_code, model=self._model, device=self._device
            )
        return self._pipelines[lang_code]

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio from phoneme tokens.

        Args:
            tokens: Input phoneme tokens to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Memory management for GPU
            if self._device == "cuda":
                if self._check_memory():
                    self._clear_memory()

            # Handle voice input with caching
            voice_tensor: torch.Tensor
            voice_name: str
            if isinstance(voice, tuple):
                voice_name, voice_data = voice
                if isinstance(voice_data, str):
                    # Use cached voice tensor
                    voice_tensor = await self._get_cached_voice_tensor(voice_name, voice_data)
                else:
                    # Use tensor directly, ensuring it's on the correct device
                    voice_tensor = voice_data.to(self._device)
                    # Cache the tensor for future use
                    self._voice_cache[voice_name] = voice_tensor
            else:
                voice_name = os.path.splitext(os.path.basename(voice))[0]
                # Use cached voice tensor
                voice_tensor = await self._get_cached_voice_tensor(voice_name, voice)

            # Use provided lang_code, settings voice code override, or first letter of voice name
            if lang_code:  # api is given priority
                pipeline_lang_code = lang_code
            elif settings.default_voice_code:  # settings is next priority
                pipeline_lang_code = settings.default_voice_code
            else:  # voice name is default/fallback
                pipeline_lang_code = voice_name[0].lower()

            pipeline = self._get_pipeline(pipeline_lang_code)

            logger.debug(
                f"Generating audio from tokens with lang_code '{pipeline_lang_code}': '{tokens[:100]}{'...' if len(tokens) > 100 else ''}'"
            )
            for result in pipeline.generate_from_tokens(
                tokens=tokens, voice=voice_tensor, speed=speed, model=self._model
            ):
                if result.audio is not None:
                    logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                    yield result.audio.numpy()
                else:
                    logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate_from_tokens(
                    tokens, voice, speed, lang_code
                ):
                    yield chunk
            raise

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio using model.

        Args:
            text: Input text to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        try:
            # Memory management for GPU
            if self._device == "cuda":
                if self._check_memory():
                    self._clear_memory()

            # Handle voice input with caching
            voice_tensor: torch.Tensor
            voice_name: str
            if isinstance(voice, tuple):
                voice_name, voice_data = voice
                if isinstance(voice_data, str):
                    # Use cached voice tensor
                    voice_tensor = await self._get_cached_voice_tensor(voice_name, voice_data)
                else:
                    # Use tensor directly, ensuring it's on the correct device
                    voice_tensor = voice_data.to(self._device)
                    # Cache the tensor for future use
                    self._voice_cache[voice_name] = voice_tensor
            else:
                voice_name = os.path.splitext(os.path.basename(voice))[0]
                # Use cached voice tensor
                voice_tensor = await self._get_cached_voice_tensor(voice_name, voice)

            # Use provided lang_code, settings voice code override, or first letter of voice name
            pipeline_lang_code = (
                lang_code
                if lang_code
                else (
                    settings.default_voice_code
                    if settings.default_voice_code
                    else voice_name[0].lower()
                )
            )
            pipeline = self._get_pipeline(pipeline_lang_code)

            logger.debug(
                f"Generating audio for text with lang_code '{pipeline_lang_code}': '{text[:100]}{'...' if len(text) > 100 else ''}'"
            )
            for result in pipeline(
                text, voice=voice_tensor, speed=speed, model=self._model
            ):
                if result.audio is not None:
                    logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                    word_timestamps = None
                    if (
                        return_timestamps
                        and hasattr(result, "tokens")
                        and result.tokens
                    ):
                        word_timestamps = []
                        current_offset = 0.0
                        logger.debug(
                            f"Processing chunk timestamps with {len(result.tokens)} tokens"
                        )
                        if result.pred_dur is not None:
                            try:
                                # Add timestamps with offset
                                for token in result.tokens:
                                    if not all(
                                        hasattr(token, attr)
                                        for attr in [
                                            "text",
                                            "start_ts",
                                            "end_ts",
                                        ]
                                    ):
                                        continue
                                    if not token.text or not token.text.strip():
                                        continue

                                    start_time = float(token.start_ts) + current_offset
                                    end_time = float(token.end_ts) + current_offset
                                    word_timestamps.append(
                                        WordTimestamp(
                                            word=str(token.text).strip(),
                                            start_time=start_time,
                                            end_time=end_time,
                                        )
                                    )
                                    logger.debug(
                                        f"Added timestamp for word '{token.text}': {start_time:.3f}s - {end_time:.3f}s"
                                    )

                            except Exception as e:
                                logger.error(
                                    f"Failed to process timestamps for chunk: {e}"
                                )

                    yield AudioChunk(
                        result.audio.numpy(), word_timestamps=word_timestamps
                    )
                else:
                    logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate(text, voice, speed, lang_code):
                    yield chunk
            raise

    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold."""
        if self._device == "cuda":
            memory_gb = torch.cuda.memory_allocated() / 1e9
            return memory_gb > model_config.pytorch_gpu.memory_threshold
        # MPS doesn't provide memory management APIs
        return False

    def _clear_memory(self) -> None:
        """Clear device memory."""
        if self._device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self._device == "mps":
            # Empty cache if available (future-proofing)
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        for pipeline in self._pipelines.values():
            del pipeline
        self._pipelines.clear()
        # Clear voice cache
        self._voice_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device

    async def _get_cached_voice_tensor(self, voice_name: str, voice_path: str) -> torch.Tensor:
        """Get voice tensor from cache or load and cache it.
        
        Args:
            voice_name: Name of the voice for cache key
            voice_path: Path to the voice file
            
        Returns:
            Voice tensor on the correct device
        """
        # Check if voice is already cached
        if voice_name in self._voice_cache:
            logger.debug(f"Using cached voice tensor for: {voice_name}")
            return self._voice_cache[voice_name]
        
        # If cache is full, remove oldest entry (simple FIFO for now)
        if len(self._voice_cache) >= self._voice_cache_size:
            oldest_key = next(iter(self._voice_cache))
            logger.debug(f"Voice cache full, removing: {oldest_key}")
            del self._voice_cache[oldest_key]
        
        # Load voice tensor asynchronously
        voice_tensor = await paths.load_voice_tensor(voice_path, device=self._device)
        
        # Cache the tensor
        self._voice_cache[voice_name] = voice_tensor
        logger.debug(f"Cached voice tensor for: {voice_name}")
        
        return voice_tensor

    def get_voice_cache_info(self) -> Dict[str, Union[int, float]]:
        """Get voice cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_size_mb = 0.0
        for tensor in self._voice_cache.values():
            # Estimate tensor size in MB
            cache_size_mb += tensor.numel() * tensor.element_size() / (1024 * 1024)
        
        return {
            "cached_voices": len(self._voice_cache),
            "max_cache_size": self._voice_cache_size,
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_utilization": len(self._voice_cache) / self._voice_cache_size
        }

    def clear_voice_cache(self) -> None:
        """Clear the voice tensor cache."""
        logger.info(f"Clearing voice cache with {len(self._voice_cache)} entries")
        self._voice_cache.clear()
