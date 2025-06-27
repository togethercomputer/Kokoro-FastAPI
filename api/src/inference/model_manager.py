"""Kokoro V1 model management."""

import multiprocessing as mp
import asyncio
from typing import Optional

from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1
from .voice_manager import VoiceManager



async def subprocess_async_run(input_queue: mp.Queue, output_queue: mp.Queue, config: ModelConfig, voice_manager: VoiceManager):
    manager = ModelManager(config)
    _ = await manager.initialize_with_warmup(voice_manager)
    while True:
        input_data = input_queue.get()
        if input_data is None:
            # Signal end of processing
            output_queue.put(None)
            break
        args, kwargs = input_data
        chunk_text = args[0]
        speed = kwargs.get("speed", 1.0)
        output_format = kwargs.pop("output_format", "mp3")
        volume_multiplier = kwargs.pop("volume_multiplier", 1.0)
        
        try:
            async for result in manager.generate(*args, **kwargs):
                logger.info("start put...")
                output_queue.put((chunk_text, result, speed, output_format, volume_multiplier))
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            # Still put None to signal end of this chunk
            output_queue.put(None)
        
        # Signal end of this chunk
        output_queue.put(None)


def subprocess_run(input_queue: mp.Queue, output_queue: mp.Queue, config: ModelConfig, voice_manager: VoiceManager):
    asyncio.run(subprocess_async_run(input_queue, output_queue, config, voice_manager))


async def post_process(input_queue: mp.Queue, output_queue: mp.Queue):
    from ..services.streaming_audio_writer import StreamingAudioWriter
    from ..services.audio import AudioService, AudioNormalizer
    writers = {}
    is_last = False
    normalizer = AudioNormalizer()
    while True:
        input_data = input_queue.get()
        logger.info("post process get...")
        if input_data is None:
            # End of all processing
            output_queue.put(None)
            continue
        chunk_text, chunk_data, speed, output_format, volume_multiplier = input_data
        if output_format not in writers:
            writers[output_format] = StreamingAudioWriter(output_format, sample_rate=24000)
        writer = writers[output_format]
        chunk_data.audio*=volume_multiplier
        # For streaming, convert to bytes
        if output_format:
            try:
                chunk_data = await AudioService.convert_audio(
                    chunk_data,
                    output_format,
                    writer,
                    speed,
                    chunk_text,
                    is_last_chunk=is_last,
                    normalizer=normalizer,
                )
                output_queue.put(chunk_data)
            except Exception as e:
                logger.error(f"Failed to convert audio: {str(e)}")
        else:
            chunk_data = AudioService.trim_audio(
                chunk_data, chunk_text, speed, is_last, normalizer
            )
            output_queue.put(chunk_data)


def post_process_async(input_queue: mp.Queue, output_queue: mp.Queue):
    asyncio.run(post_process(input_queue, output_queue))


class ModelManager:
    """Manages Kokoro V1 model loading and inference."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        self._backend: Optional[KokoroV1] = None  # Explicitly type as KokoroV1
        self._device: Optional[str] = None

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        """Initialize Kokoro V1 backend."""
        try:
            self._device = self._determine_device()
            logger.info(f"Initializing Kokoro V1 on {self._device}")
            self._backend = KokoroV1()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kokoro V1: {e}")

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        """Initialize and warm up model.

        Args:
            voice_manager: Voice manager instance for warmup

        Returns:
            Tuple of (device, backend type, voice count)

        Raises:
            RuntimeError: If initialization fails
        """
        import time

        start = time.perf_counter()

        try:
            # Initialize backend
            await self.initialize()

            # Load model
            model_path = self._config.pytorch_kokoro_v1_file
            await self.load_model(model_path)

            # Use paths module to get voice path
            try:
                voices = await paths.list_voices()
                voice_path = await paths.get_voice_path(settings.default_voice)

                # Warm up with short text
                warmup_text = "Warmup text for initialization."
                # Use default voice name for warmup
                voice_name = settings.default_voice
                logger.debug(f"Using default voice '{voice_name}' for warmup")
                async for _ in self.generate(warmup_text, (voice_name, voice_path)):
                    pass
            except Exception as e:
                raise RuntimeError(f"Failed to get default voice: {e}")

            ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"Warmup completed in {ms}ms")

            return self._device, "kokoro_v1", len(voices)
        except FileNotFoundError as e:
            logger.error("""
Model files not found! You need to download the Kokoro V1 model:

1. Download model using the script:
   python docker/scripts/download_model.py --output api/src/models/v1_0

2. Or set environment variable in docker-compose:
   DOWNLOAD_MODEL=true
""")
            exit(0)
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    def get_backend(self) -> BaseModelBackend:
        """Get initialized backend.

        Returns:
            Initialized backend instance

        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        return self._backend

    async def load_model(self, path: str) -> None:
        """Load model using initialized backend.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If loading fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            await self._backend.load_model(path)
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    async def generate(self, *args, **kwargs):
        """Generate audio using initialized backend.

        Raises:
            RuntimeError: If generation fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            async for chunk in self._backend.generate(*args, **kwargs):
                if settings.default_volume_multiplier != 1.0:
                    chunk.audio *= settings.default_volume_multiplier
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def unload_all(self) -> None:
        """Unload model and free resources."""
        if self._backend:
            self._backend.unload()
            self._backend = None

    @property
    def current_backend(self) -> str:
        """Get current backend type."""
        return "kokoro_v1"


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override

    Returns:
        ModelManager instance
    """
    if ModelManager._instance is None:
        ModelManager._instance = SubprocessManager(config or model_config)
    return ModelManager._instance


class SubprocessManager(ModelManager):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.ctx = mp.get_context("spawn")
        self._input_queue = self.ctx.Queue()
        self._output_queue = self.ctx.Queue()
        self._post_process_queue = self.ctx.Queue()
    
    async def initialize_with_warmup(self, voice_manager: VoiceManager):
        self._process = self.ctx.Process(target=subprocess_run, args=(self._input_queue, self._post_process_queue, self._config, voice_manager))
        self._process.start()
        self._post_process = self.ctx.Process(target=post_process_async, args=(self._post_process_queue, self._output_queue))
        self._post_process.start()
        return "cpu", "kokoro_v1", 0
    
    async def generate(self, *args, **kwargs):
        self._input_queue.put((args, kwargs))
        while True:
            # Use asyncio.to_thread to run the blocking get() in a thread pool
            rst = await asyncio.to_thread(self._output_queue.get)
            if rst:
                yield rst
            else:
                break
    
    def put_chunk(self, args, kwargs):
        """Put a chunk into the input queue for processing."""
        self._input_queue.put((args, kwargs))
    
    async def get_output(self):
        """Get output from the output queue."""
        return await asyncio.to_thread(self._output_queue.get)
    
    def signal_end_of_input(self):
        """Signal that no more input will be provided."""
        self._input_queue.put(None)
    
    def get_backend(self):
        return KokoroV1()
