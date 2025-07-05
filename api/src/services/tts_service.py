"""TTS service using model and voice managers."""

import asyncio
import os
import queue
import re
import tempfile
import time
import threading
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from kokoro import KPipeline
from loguru import logger

from ..core.config import settings
from ..inference.base import AudioChunk
from ..inference.kokoro_v1 import KokoroV1
from ..inference.voice_manager import get_manager as get_voice_manager
from ..structures.schemas import NormalizationOptions
from .audio import AudioNormalizer, AudioService
from .streaming_audio_writer import StreamingAudioWriter
from .text_processing import tokenize
from .text_processing.text_processor import process_text_chunk, smart_split


class TTSService:
    """Text-to-speech service."""

    # Limit concurrent chunk processing
    _chunk_semaphore = asyncio.Semaphore(4)

    def __init__(self, output_dir: str = None):
        """Initialize service."""
        self.output_dir = output_dir
        self.model_manager = None
        self._voice_manager = None
        self.output_queues: Dict[int, asyncio.Queue] = {}
        self.num_requests = 0
        self._check_loop_thread = None
        self.input_queue = mp.get_context("spawn").Queue()
        self.output_queue = mp.get_context("spawn").Queue()
    
    def check_loop(self):
        """Run in a separate thread to handle blocking queue operations."""
        while True:
            try:
                # Block indefinitely - maximum throughput, no timeout overhead
                request_id, chunk_data = self.output_queue.get()
                self.output_queues[request_id].put_nowait(chunk_data)
            except Exception as e:
                logger.error(f"Error in check_loop thread: {e}")
                break

    def start_check_loop(self):
        """Start the check loop thread."""
        if self._check_loop_thread is None:
            self._check_loop_thread = threading.Thread(target=self.check_loop, daemon=True)
            self._check_loop_thread.start()
            logger.info("Check loop thread started")

    @classmethod
    async def create(cls, output_dir: str = None) -> "TTSService":
        """Create and initialize TTSService instance."""
        service = cls(output_dir)
        from ..inference.mp_model_manager import get_manager as get_mp_model_manager
        service.model_manager = await get_mp_model_manager(input_queue=service.input_queue, output_queue=service.output_queue)
        service._voice_manager = await get_voice_manager()
        # Start the check loop thread
        await service.model_manager.initialize()
        service.start_check_loop()
        return service

    async def _process_chunk(
        self,
        chunk_text: str,
        tokens: List[int],
        voice_name: str,
        voice_path: str,
        speed: float,
        writer: StreamingAudioWriter,
        output_format: Optional[str] = None,
        is_first: bool = False,
        is_last: bool = False,
        volume_multiplier: Optional[float] = 1.0,
        normalizer: Optional[AudioNormalizer] = None,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Process tokens into audio."""
        async with self._chunk_semaphore:
            try:
                # Handle stream finalization
                if is_last:
                    # Skip format conversion for raw audio mode
                    if not output_format:
                        yield AudioChunk(np.array([], dtype=np.int16), output=b"")
                        return
                    chunk_data = await AudioService.convert_audio(
                        AudioChunk(
                            np.array([], dtype=np.float32)
                        ),  # Dummy data for type checking
                        output_format,
                        writer,
                        speed,
                        "",
                        normalizer=normalizer,
                        is_last_chunk=True,
                    )
                    yield chunk_data
                    return

                # Skip empty chunks
                if not tokens and not chunk_text:
                    return

                # Get backend
                backend = self.model_manager.get_backend()

                # Generate audio using pre-warmed model
                if isinstance(backend, KokoroV1):
                    chunk_index = 0
                    # For Kokoro V1, pass text and voice info with lang_code
                    async for chunk_data in self.model_manager.generate(
                        chunk_text,
                        (voice_name, voice_path),
                        speed=speed,
                        lang_code=lang_code,
                        return_timestamps=return_timestamps,
                    ):
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
                                yield chunk_data
                            except Exception as e:
                                logger.error(f"Failed to convert audio: {str(e)}")
                        else:
                            chunk_data = AudioService.trim_audio(
                                chunk_data, chunk_text, speed, is_last, normalizer
                            )
                            yield chunk_data
                        chunk_index += 1
            except Exception as e:
                logger.error(f"Failed to process tokens: {str(e)}")

    async def _load_voice_from_path(self, path: str, weight: float):
        # Check if the path is None and raise a ValueError if it is not
        if not path:
            raise ValueError(f"Voice not found at path: {path}")

        logger.debug(f"Loading voice tensor from path: {path}")
        return torch.load(path, map_location="cpu") * weight

    async def _get_voices_path(self, voice: str) -> Tuple[str, str]:
        """Get voice path, handling combined voices.

        Args:
            voice: Voice name or combined voice names (e.g., 'af_jadzia+af_jessica')

        Returns:
            Tuple of (voice name to use, voice path to use)

        Raises:
            RuntimeError: If voice not found
        """
        try:
            # Split the voice on + and - and ensure that they get added to the list eg: hi+bob = ["hi","+","bob"]
            split_voice = re.split(r"([-+])", voice)

            # If it is only once voice there is no point in loading it up, doing nothing with it, then saving it
            if len(split_voice) == 1:
                # Since its a single voice the only time that the weight would matter is if voice_weight_normalization is off
                if (
                    "(" not in voice and ")" not in voice
                ) or settings.voice_weight_normalization == True:
                    path = await self._voice_manager.get_voice_path(voice)
                    if not path:
                        raise RuntimeError(f"Voice not found: {voice}")
                    logger.debug(f"Using single voice path: {path}")
                    return voice, path

            total_weight = 0

            for voice_index in range(0, len(split_voice), 2):
                voice_object = split_voice[voice_index]

                if "(" in voice_object and ")" in voice_object:
                    voice_name = voice_object.split("(")[0].strip()
                    voice_weight = float(voice_object.split("(")[1].split(")")[0])
                else:
                    voice_name = voice_object
                    voice_weight = 1

                total_weight += voice_weight
                split_voice[voice_index] = (voice_name, voice_weight)

            # If voice_weight_normalization is false prevent normalizing the weights by setting the total_weight to 1 so it divides each weight by 1
            if settings.voice_weight_normalization == False:
                total_weight = 1

            # Load the first voice as the starting point for voices to be combined onto
            path = await self._voice_manager.get_voice_path(split_voice[0][0])
            combined_tensor = await self._load_voice_from_path(
                path, split_voice[0][1] / total_weight
            )

            # Loop through each + or - in split_voice so they can be applied to combined voice
            for operation_index in range(1, len(split_voice) - 1, 2):
                # Get the voice path of the voice 1 index ahead of the operator
                path = await self._voice_manager.get_voice_path(
                    split_voice[operation_index + 1][0]
                )
                voice_tensor = await self._load_voice_from_path(
                    path, split_voice[operation_index + 1][1] / total_weight
                )

                # Either add or subtract the voice from the current combined voice
                if split_voice[operation_index] == "+":
                    combined_tensor += voice_tensor
                else:
                    combined_tensor -= voice_tensor

            # Save the new combined voice so it can be loaded latter
            temp_dir = tempfile.gettempdir()
            combined_path = os.path.join(temp_dir, f"{voice}.pt")
            logger.debug(f"Saving combined voice to: {combined_path}")
            torch.save(combined_tensor, combined_path)
            return voice, combined_path
        except Exception as e:
            logger.error(f"Failed to get voice path: {e}")
            raise

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter,
        speed: float = 1.0,
        output_format: str = "wav",
        lang_code: Optional[str] = None,
        volume_multiplier: Optional[float] = 1.0,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        request_id = self.num_requests
        print(f"request_id: {request_id}")
        self.num_requests += 1
        self.output_queues[request_id] = asyncio.Queue()

        """Generate and stream audio chunks."""
        # Get voice path, handling combined voices
        voice_name, voice_path = await self._get_voices_path(voice)
        logger.debug(f"Using voice path: {voice_path}")

        # Use provided lang_code or determine from voice name
        pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
        logger.info(
            f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in audio stream"
        )
        self.input_queue.put((request_id, text, voice_name, speed, output_format, pipeline_lang_code, volume_multiplier, normalization_options, return_timestamps))

        try:
            items = 0
            while True:
                # Get item from the async queue for this request
                chunk_data = await self.output_queues[request_id].get()
                
                # Check if this is a sentinel value to stop the generator
                if chunk_data is None:
                    break
                items += 1
                yield chunk_data
                await asyncio.sleep(0)

        finally:
            # Clean up the queue when done
            if request_id in self.output_queues:
                del self.output_queues[request_id]


    async def generate_audio(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter,
        speed: float = 1.0,
        return_timestamps: bool = False,
        volume_multiplier: Optional[float] = 1.0,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        lang_code: Optional[str] = None,
    ) -> AudioChunk:
        """Generate complete audio for text using streaming internally."""
        audio_data_chunks = []

        try:
            async for audio_stream_data in self.generate_audio_stream(
                text,
                voice,
                writer,
                speed=speed,
                volume_multiplier=volume_multiplier,
                normalization_options=normalization_options,
                return_timestamps=return_timestamps,
                lang_code=lang_code,
                output_format=None,
            ):
                if len(audio_stream_data.audio) > 0:
                    audio_data_chunks.append(audio_stream_data)

            combined_audio_data = AudioChunk.combine(audio_data_chunks)
            return combined_audio_data
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def combine_voices(self, voices: List[str]) -> torch.Tensor:
        """Combine multiple voices.

        Returns:
            Combined voice tensor
        """

        return await self._voice_manager.combine_voices(voices)

    async def list_voices(self) -> List[str]:
        """List available voices."""
        return await self._voice_manager.list_voices()

    async def generate_from_phonemes(
        self,
        phonemes: str,
        voice: str,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """Generate audio directly from phonemes.

        Args:
            phonemes: Phonemes in Kokoro format
            voice: Voice name
            speed: Speed multiplier
            lang_code: Optional language code override

        Returns:
            Tuple of (audio array, processing time)
        """
        start_time = time.time()
        try:
            # Get backend and voice path
            backend = self.model_manager.get_backend()
            voice_name, voice_path = await self._get_voices_path(voice)

            if isinstance(backend, KokoroV1):
                # For Kokoro V1, use generate_from_tokens with raw phonemes
                result = None
                # Use provided lang_code or determine from voice name
                pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
                logger.info(
                    f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in phoneme pipeline"
                )

                try:
                    # Use backend's pipeline management
                    for r in backend._get_pipeline(
                        pipeline_lang_code
                    ).generate_from_tokens(
                        tokens=phonemes,  # Pass raw phonemes string
                        voice=voice_path,
                        speed=speed,
                    ):
                        if r.audio is not None:
                            result = r
                            break
                except Exception as e:
                    logger.error(f"Failed to generate from phonemes: {e}")
                    raise RuntimeError(f"Phoneme generation failed: {e}")

                if result is None or result.audio is None:
                    raise ValueError("No audio generated")

                processing_time = time.time() - start_time
                return result.audio.numpy(), processing_time
            else:
                raise ValueError(
                    "Phoneme generation only supported with Kokoro V1 backend"
                )

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise
