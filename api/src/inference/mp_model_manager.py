"""Kokoro V1 model management."""

import asyncio
import os
import re
import torch
import numpy as np
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Dict, List
from loguru import logger
from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1
from kokoro import KPipeline, KModel
from .request_queue import FairRequestQueue
from ..services.audio import AudioService, AudioNormalizer
from ..inference.base import AudioChunk
from ..services.text_processing import smart_split
from ..inference.voice_manager import get_manager as get_voice_manager
from ..services.streaming_audio_writer import StreamingAudioWriter


@dataclass
class PostProcessingData:
    request_id: str
    audio: Optional[torch.Tensor] = None
    pred_dur: Optional[torch.Tensor] = None
    pause_duration_s: float = 0.0
    output_format: Optional[str] = None
    volume_multiplier: Optional[float] = None
    speed: float = 1.0
    data_type: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class ModelInferenceData:
    request_id: str
    input_ids: torch.Tensor
    voice_list: List[str]
    voice_weights: List[float]
    speed: float
    data_type: str
    metadata: Optional[Dict] = None


@dataclass
class StopSignal:
    request_id: str


class SimpleKPipeline(KPipeline):
    """Simple Kokoro V1 pipeline for preprocessing"""

    def __init__(self, lang_code: str):
        super().__init__(lang_code)
        self.device = settings.get_device()

    def preprocess_generate(self, text, split_pattern: Optional[str] = r'\n+',):
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]\
        
        for graphemes_index, graphemes in enumerate(text):
            if not graphemes.strip():
                continue
            # English processing (unchanged)
            if self.lang_code in 'ab':
                logger.debug(f"Processing English text: {graphemes[:50]}{'...' if len(graphemes) > 50 else ''}")
                _, tokens = self.g2p(graphemes)
                for gs, ps, tks in self.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logger.warning(f"Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'")
                        ps = ps[:510]
                    input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), ps)))
                    # not sure if cuda tensor can be transferred in a queue here, may need torch multiprocessing queue
                    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device, non_blocking=True)
                    yield (gs, input_ids, tks, graphemes_index)
                    # yield self.Result(graphemes=gs, phonemes=ps, tokens=tks, output=output, text_index=graphemes_index)
            
            # Non-English processing with chunking
            else:
                # Split long text into smaller chunks (roughly 400 characters each)
                # Using sentence boundaries when possible
                chunk_size = 400
                chunks = []
                
                # Try to split on sentence boundaries first
                sentences = re.split(r'([.!?]+)', graphemes)
                current_chunk = ""
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    # Add the punctuation back if it exists
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]
                        
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence

                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If no chunks were created (no sentence boundaries), fall back to character-based chunking
                if not chunks:
                    chunks = [graphemes[i:i+chunk_size] for i in range(0, len(graphemes), chunk_size)]
                
                # Process each chunk
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                        
                    ps, _ = self.g2p(chunk)
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logger.warning(f'Truncating len(ps) == {len(ps)} > 510')
                        ps = ps[:510]

                    yield None, ps, None, graphemes_index


class SubProcessModelManager:
    """Manages Kokoro V1 preprocessing, model loading, inference and postprocessing in a subprocess."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None, input_queue: mp.Queue = None, output_queue: mp.Queue = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        self._backend: Optional[KokoroV1] = None  # Explicitly type as KokoroV1
        self._device: Optional[str] = None
        ctx = mp.get_context("spawn")
        self.postprocessing_queue = ctx.Queue()
        self.model_queue = ctx.Queue()
        self.fair_queue = FairRequestQueue()
        self.input_queue = input_queue
        self.result_queue = output_queue
    
    def preprocess_fn(self):
        pipelines = {}
        while True:
            request_id, text, voice_name, speed, output_format, lang_code, volume_multiplier, normalization_options, return_timestamps = self.input_queue.get_nowait()
            asyncio.create_task(self.preprocessing(pipelines, text, lang_code, normalization_options, output_format, request_id, voice_name, speed, volume_multiplier))

    async def preprocessing(self, pipelines, text, lang_code, normalization_options, output_format, request_id, voice_name, speed, volume_multiplier):
        chunk_index = 0
        voice_list, voice_weights = self.get_voice_list(voice_name)
        async for chunk_text, tokens, pause_duration_s in smart_split(
            text,
            lang_code=lang_code,
            normalization_options=normalization_options,
        ):
            if pause_duration_s is not None and pause_duration_s > 0:
                # pause, send to postprocessing queue directly
                data_type = "pause"
                data = PostProcessingData(request_id, None, None, pause_duration_s, output_format, data_type=data_type)
                self.postprocessing_queue.put(data)
                current_offset += pause_duration_s
                chunk_index += 1  # Count pause as a yielded chunk
            elif tokens or chunk_text.strip():
                data_type = "text"
                if lang_code not in pipelines:
                    pipelines[lang_code] = SimpleKPipeline(lang_code)
                
                pipeline = pipelines[lang_code]
                for gs, input_ids, tks, graphemes_index in pipeline.preprocess_generate(chunk_text):
                    metadata = dict(
                        gs=gs,
                        tokens=tks,
                        graphemes_index=graphemes_index,
                        volume_multiplier=volume_multiplier,
                    )
                    data = ModelInferenceData(request_id, input_ids, voice_list, voice_weights, speed, data_type, metadata)
                    self.model_queue.put(data)
                chunk_index += 1  # Increment chunk index after processing text
            
            if chunk_index > 0:
                data_type = "final"
                data = PostProcessingData(request_id, None, None, 0.0, output_format, data_type=data_type)
                self.postprocessing_queue.put(data)
            await asyncio.sleep(0)

    
    def model_inference_fn(self):
        voice_manager = get_voice_manager()
        model_path = self._config.pytorch_kokoro_v1_file
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        self.model = KModel(config=config_path, model=model_path).eval()
        self.model.to(settings.get_device())
        while True:
            self.model_inference(voice_manager)
    
    @staticmethod
    def _get_voice_tensor(voice_name, voice_manager)->torch.Tensor:
        if voice_name not in voice_manager._voices:
            voice_tensor = voice_manager.load_voice(voice_name)
        else:
            voice_tensor = voice_manager._voices[voice_name]
        return voice_tensor
    
    def get_voice_list(self, voice_name)->tuple[List[str], List[float]]:
        split_voices = re.split(r"([-+])", voice_name)
        if len(split_voices) == 1:
            if (
                "(" not in voice_name and ")" not in voice_name
            ) or settings.voice_weight_normalization == True:
                return [voice_name], [1]
        
        total_weight = 0
        voice_list = []
        voice_weights = []
        for voice_index in range(0, len(split_voices), 2):
            voice_object = split_voices[voice_index]

            if "(" in voice_object and ")" in voice_object:
                voice_name = voice_object.split("(")[0].strip()
                voice_weight = float(voice_object.split("(")[1].split(")")[0])
            else:
                voice_name = voice_object
                voice_weight = 1.0

            total_weight += voice_weight
            split_voices[voice_index] = (voice_name, voice_weight)
        
        if settings.voice_weight_normalization == False:
            total_weight = 1.0
        
        voice_list.append(split_voices[0][0])
        voice_weights.append(split_voices[0][1] / total_weight)
        for operation_index in range(1, len(split_voices) - 1, 2):
            operation = split_voices[operation_index]
            voice_list.append(split_voices[operation_index + 1][0])
            if operation == "+":
                voice_weights.append(split_voices[operation_index + 1][1] / total_weight)
            elif operation == "-":
                voice_weights.append(-split_voices[operation_index + 1][1] / total_weight)
        
        return voice_list, voice_weights

    def load_voice(self, voice_list, voice_weights, voice_manager)->torch.Tensor:
        voice_tensors = []
        for voice_name, voice_weight in zip(voice_list, voice_weights):
            voice_tensor = self._get_voice_tensor(voice_name, voice_manager)
            voice_tensors.append(voice_tensor * voice_weight)
        return sum(voice_tensors)

    def model_inference(self, voice_manager):
        # get all requests from model queue, sort them in a priority queue
        while not self.model_queue.empty():
            request = self.model_queue.get_nowait()
            self.fair_queue.add_request(request)

        while not self.fair_queue.empty():
            data = self.fair_queue.pop_request()
            voice_tensor = self.load_voice(data.voice_list, data.voice_weights, voice_manager)
            audio, pred_dur = self.model.forward_with_tokens(data.input_ids, voice_tensor, data.speed)
            del data.input_ids
            post_data = PostProcessingData(data.request_id, audio, pred_dur, 0.0, data.output_format, data_type=data.data_type, metadata=data.metadata)
            self.postprocessing_queue.put(post_data)

    def postprocessing_fn(self):
        writers = {}
        offsets_mapping = {}
        while True:
            self.postprocessing(offsets_mapping, writers)

    def postprocessing(self, offsets_mapping: Dict, writers: Dict):
        stream_normalizer = AudioNormalizer()
        while not self.postprocessing_queue.empty():
            output = self.postprocessing_queue.get()
            data_type = output.data_type
            output_format = output.output_format
            if not output_format in writers:
                writers[output_format] = StreamingAudioWriter(output_format, sample_rate=24000)
            writer = writers[output_format]
            request_id = output.request_id
            speed = output.speed
            chunk_text = output.chunk_text
            pause_duration_s = output.pause_duration_s
            if request_id not in offsets_mapping:
                offsets_mapping[request_id] = 0.0
            if data_type == "pause":
                silence_samples = int(pause_duration_s * 24000)
                silence_audio = np.zeros(silence_samples, dtype=np.int16)
                pause_chunk = AudioChunk(audio=silence_audio, word_timestamps=[]) # Empty timestamps for silence
                chunk_data = AudioService.convert_audio(
                    pause_chunk, output_format, writer, speed=speed, chunk_text="",
                    is_last_chunk=False, trim_audio=False, normalizer=stream_normalizer,
                )
                offsets_mapping[request_id] += pause_duration_s
            elif data_type == "text":
                chunk_data *= output.metadata["volume_multiplier"]
                chunk_data = AudioService.convert_audio(
                    chunk_data,
                    output_format,
                    writer,
                    speed,
                    chunk_text,
                    is_last_chunk=False,
                    normalizer=stream_normalizer,
                )
                if chunk_data.word_timestamps is not None:
                    for timestamp in chunk_data.word_timestamps:
                        timestamp.start_time += offsets_mapping[request_id]
                        timestamp.end_time += offsets_mapping[request_id]

                # Update offset based on the actual duration of the generated audio chunk
                chunk_duration = 0
                if chunk_data.audio is not None and len(chunk_data.audio) > 0:
                    chunk_duration = len(chunk_data.audio) / 24000
                    offsets_mapping[request_id] += chunk_duration
            elif data_type == "final":
                chunk_data = AudioService.convert_audio(
                    AudioChunk(
                        np.array([], dtype=np.float32)
                    ),  # Dummy data for type checking
                    output_format,
                    writer,
                    speed,
                    "",
                    normalizer=stream_normalizer,
                    is_last_chunk=True,
                )
                offsets_mapping.pop(request_id)
            
            self.result_queue.put((request_id, chunk_data))
            if request_id not in offsets_mapping:
                self.result_queue.put((request_id, None))

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        """Initialize Kokoro V1 backend."""
        try:
            self._device = self._determine_device()
            logger.info(f"Initializing Kokoro V1 on {self._device}")
            self._backend = KokoroV1()
            ctx = mp.get_context("spawn")
            self.preprocess_process = ctx.Process(target=self.preprocess_fn)
            self.model_inference_process = ctx.Process(target=self.model_inference_fn)
            self.postprocessing_process = ctx.Process(target=self.postprocessing_fn)
            self.preprocess_process.start()
            self.model_inference_process.start()
            self.postprocessing_process.start()

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

            # Use paths module to get voice path
            try:
                voices = await paths.list_voices()
                # Warm up with short text
                warmup_text = "Warmup text for initialization."
                # Use default voice name for warmup
                voice_name = settings.default_voice
                logger.debug(f"Using default voice '{voice_name}' for warmup")
                self.input_queue.put((0, warmup_text, voice_name, 1.0, "wav", "en", 1.0, None, False))
                while True:
                    _, chunk_data = self.result_queue.get()
                    if chunk_data is None:
                        break
                    else:
                        pass
            except Exception as e:
                raise RuntimeError(f"Failed to get default voice: {e}")

            ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"Warmup completed in {ms}ms")

            return self._device, "kokoro_v1", len(voices)
        except FileNotFoundError as e:
            logger.error("Model files not found!")
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

    def unload_all(self) -> None:
        """Unload model and free resources."""
        if self._backend:
            self._backend.unload()
            self._backend = None

    @property
    def current_backend(self) -> str:
        """Get current backend type."""
        return "kokoro_v1"


async def get_manager(config: Optional[ModelConfig] = None) -> SubProcessModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override

    Returns:
        ModelManager instance
    """
    if SubProcessModelManager._instance is None:
        SubProcessModelManager._instance = SubProcessModelManager(config)
    return SubProcessModelManager._instance
