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
from ..inference.voice_manager import get_manager_sync, VoiceManager
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
    input_ids: Optional[torch.Tensor]
    voice_list: List[str]
    voice_weights: List[float]
    speed: float
    output_format: str
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
        config_path = "api/src/models/v1_0/config.json"
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        self.vocab = config["vocab"]

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
                    yield (gs, ps, input_ids, tks, graphemes_index)
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


def preprocess_worker(input_queue, model_queue, postprocessing_queue):
    """Standalone preprocessing function for multiprocessing."""
    pipelines = {}
    while True:
        try:
            request_id, text, voice_name, speed, output_format, lang_code, volume_multiplier, normalization_options, return_timestamps = input_queue.get()
            
            print("get input")
            # Create event loop for this process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run preprocessing
            loop.run_until_complete(
                preprocessing_task(pipelines, text, lang_code, normalization_options, output_format, 
                                 request_id, voice_name, speed, volume_multiplier, model_queue, postprocessing_queue)
            )
            
        except Exception as e:
            raise e
            logger.error(f"Error in preprocess_worker: {e}")
            continue


async def preprocessing_task(pipelines, text, lang_code, normalization_options, output_format, request_id, voice_name, speed, volume_multiplier, model_queue, postprocessing_queue):
    """Async preprocessing task."""
    chunk_index = 0
    voice_list, voice_weights = get_voice_list(voice_name)
    # print(text, lang_code, normalization_options, output_format, request_id, voice_name, speed, volume_multiplier)
    
    async for chunk_text, tokens, pause_duration_s in smart_split(
        text,
        lang_code=lang_code,
        normalization_options=normalization_options,
    ):
        if pause_duration_s is not None and pause_duration_s > 0:
            # pause, send to postprocessing queue directly
            data_type = "pause"
            metadata = dict(pause_duration_s=pause_duration_s)
            data = ModelInferenceData(request_id, None, [], [], 1.0, output_format, data_type, metadata)
            # data = PostProcessingData(request_id, None, None, pause_duration_s, output_format, data_type=data_type)
            model_queue.put(data)
            chunk_index += 1  # Count pause as a yielded chunk
        elif tokens or chunk_text.strip():
            data_type = "text"
            if lang_code not in pipelines:
                pipelines[lang_code] = SimpleKPipeline(lang_code)
            
            pipeline = pipelines[lang_code]
            for gs, ps, input_ids, tks, graphemes_index in pipeline.preprocess_generate(chunk_text):
                metadata = dict(
                    gs=gs,
                    ps_len=len(ps),
                    chunk_text=chunk_text,
                    volume_multiplier=volume_multiplier,
                )
                data = ModelInferenceData(request_id, input_ids, voice_list, voice_weights, speed, output_format, data_type, metadata)
                model_queue.put(data)
            chunk_index += 1  # Increment chunk index after processing text
        
        await asyncio.sleep(0)
    
    if chunk_index > 0:
        data_type = "final"
        data = ModelInferenceData(request_id, None, [], [], 1.0, output_format, data_type, {})
        model_queue.put(data)


def get_voice_list(voice_name) -> tuple[List[str], List[float]]:
    """Extract voice list and weights from voice name string."""
    split_voices = re.split(r"([-+])", voice_name)
    if len(split_voices) == 1:
        if (
            "(" not in voice_name and ")" not in voice_name
        ) or settings.voice_weight_normalization == True:
            return [voice_name], [1.0]
    
    total_weight = 0.0
    voice_list = []
    voice_weights = []
    parsed_voices = []
    
    for voice_index in range(0, len(split_voices), 2):
        voice_object = split_voices[voice_index]

        if "(" in voice_object and ")" in voice_object:
            voice_name_part = voice_object.split("(")[0].strip()
            voice_weight = float(voice_object.split("(")[1].split(")")[0])
        else:
            voice_name_part = voice_object
            voice_weight = 1.0

        total_weight += voice_weight
        parsed_voices.append((voice_name_part, voice_weight))
    
    if settings.voice_weight_normalization == False:
        total_weight = 1.0
    
    voice_list.append(parsed_voices[0][0])
    voice_weights.append(parsed_voices[0][1] / total_weight)
    
    for operation_index in range(1, len(split_voices) - 1, 2):
        operation = split_voices[operation_index]
        voice_idx = (operation_index + 1) // 2
        if voice_idx < len(parsed_voices):
            voice_list.append(parsed_voices[voice_idx][0])
            if operation == "+":
                voice_weights.append(parsed_voices[voice_idx][1] / total_weight)
            elif operation == "-":
                voice_weights.append(-parsed_voices[voice_idx][1] / total_weight)
    
    return voice_list, voice_weights


def model_inference_worker(model_queue, postprocessing_queue, config):
    """Standalone model inference function for multiprocessing."""
    voice_manager = get_manager_sync()
    model_path = config.pytorch_kokoro_v1_file
    
    # config_path = os.path.join(os.path.dirname(model_path), "config.json")
    model_path = "api/src/models/v1_0/kokoro-v1_0.pth"
    config_path = "api/src/models/v1_0/config.json"
    model = KModel(config=config_path, model=model_path).eval()
    model.to(settings.get_device())
    
    fair_queue = FairRequestQueue()
    
    while True:
        try:
            # Get all requests from model queue, sort them in a priority queue
            while not model_queue.empty():
                request = model_queue.get()
                fair_queue.add_request(request)

            while not fair_queue.empty():
                data = fair_queue.pop_request()
                if data.data_type != "text":
                    post_data = PostProcessingData(data.request_id, None, None, data.metadata.get("pause_duration_s", 0.0), data.output_format, data_type=data.data_type)
                    postprocessing_queue.put(post_data)
                    continue
                voice_tensor = load_voice(data.voice_list, data.voice_weights, voice_manager).cuda()
                
                ps_len = data.metadata.get("ps_len")
                audio, pred_dur = model.forward_with_tokens(data.input_ids, voice_tensor[ps_len-1], data.speed)
                del data.input_ids
                post_data = PostProcessingData(data.request_id, audio, pred_dur, 0.0, data.output_format, data_type=data.data_type, metadata=data.metadata)
                postprocessing_queue.put(post_data)
        except Exception as e:
            raise e
            logger.error(f"Error in model_inference_worker: {e}")
            continue


def get_voice_tensor(voice_name, voice_manager: VoiceManager) -> torch.Tensor:
    """Get voice tensor from voice manager."""
    if voice_name not in voice_manager._voices:
        voice_tensor = voice_manager.load_voice_sync(voice_name)
    else:
        voice_tensor = voice_manager._voices[voice_name]
    return voice_tensor


def load_voice(voice_list, voice_weights, voice_manager: VoiceManager) -> torch.Tensor:
    """Load and combine voice tensors based on weights."""
    voice_tensors = []
    for voice_name, voice_weight in zip(voice_list, voice_weights):
        voice_tensor = get_voice_tensor(voice_name, voice_manager)
        voice_tensors.append(voice_tensor * voice_weight)
    
    if voice_tensors:
        result = voice_tensors[0]
        for tensor in voice_tensors[1:]:
            result = result + tensor
        return result
    else:
        # Return empty tensor if no voices
        return torch.zeros(1, dtype=torch.float32)


def postprocessing_worker(postprocessing_queue, result_queue):
    """Standalone postprocessing function for multiprocessing."""
    writers = {}
    offsets_mapping = {}
    while True:
        try:
            postprocessing_task(offsets_mapping, writers, postprocessing_queue, result_queue)
        except Exception as e:
            raise e
            logger.error(f"Error in postprocessing_worker: {e}")
            continue


def postprocessing_task(offsets_mapping: Dict, writers: Dict, postprocessing_queue, result_queue):
    """Process postprocessing queue."""
    stream_normalizer = AudioNormalizer()
    while not postprocessing_queue.empty():
        output = postprocessing_queue.get()
        data_type = output.data_type
        output_format = output.output_format
       
        request_id = output.request_id
        if not request_id in writers:
            writers[request_id] = StreamingAudioWriter(output_format, sample_rate=24000)
        writer = writers[request_id]
        speed = output.speed
        if request_id not in offsets_mapping:
            offsets_mapping[request_id] = 0.0
        if data_type == "pause":
            pause_duration_s = output.pause_duration_s
            silence_samples = int(pause_duration_s * 24000)
            silence_audio = np.zeros(silence_samples, dtype=np.int16)
            pause_chunk = AudioChunk(audio=silence_audio, word_timestamps=[]) # Empty timestamps for silence
            chunk_data = AudioService.convert_audio(
                pause_chunk, output_format, writer, speed=speed, chunk_text="",
                is_last_chunk=False, trim_audio=False, normalizer=stream_normalizer,
            )
            offsets_mapping[request_id] += pause_duration_s
        elif data_type == "text":
            chunk_data = AudioChunk(audio=output.audio.cpu().numpy(), word_timestamps=[])
            del output.audio
            chunk_data.audio *= output.metadata["volume_multiplier"]
            chunk_data = AudioService.convert_audio(
                chunk_data,
                output_format,
                writer,
                speed,
                output.metadata["chunk_text"],
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
        
        result_queue.put((request_id, chunk_data))
        if request_id not in offsets_mapping:
            result_queue.put((request_id, None))


class SubProcessModelManager:
    """Manages Kokoro V1 preprocessing, model loading, inference and postprocessing in a subprocess."""

    # Singleton instance
    _instance: Optional['SubProcessModelManager'] = None

    def __init__(self, config: Optional[ModelConfig] = None, input_queue: Optional[mp.Queue] = None, output_queue: Optional[mp.Queue] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
            input_queue: Optional input queue for multiprocessing
            output_queue: Optional output queue for multiprocessing
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
            self.preprocess_process = ctx.Process(
                target=preprocess_worker,
                args=(self.input_queue, self.model_queue, self.postprocessing_queue)
            )
            self.model_inference_process = ctx.Process(
                target=model_inference_worker,
                args=(self.model_queue, self.postprocessing_queue, self._config)
            )
            self.postprocessing_process = ctx.Process(
                target=postprocessing_worker,
                args=(self.postprocessing_queue, self.result_queue)
            )
            self.preprocess_process.start()
            self.model_inference_process.start()
            self.postprocessing_process.start()

        except Exception as e:
            raise e
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
                if self.input_queue is not None and self.result_queue is not None:
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

            return self._device or "cpu", "kokoro_v1", len(voices)
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


async def get_manager(config: Optional[ModelConfig] = None, input_queue: Optional[mp.Queue] = None, output_queue: Optional[mp.Queue] = None) -> SubProcessModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override
        input_queue: Optional input queue for multiprocessing
        output_queue: Optional output queue for multiprocessing

    Returns:
        ModelManager instance
    """
    if SubProcessModelManager._instance is None:
        SubProcessModelManager._instance = SubProcessModelManager(config, input_queue, output_queue)
    return SubProcessModelManager._instance
