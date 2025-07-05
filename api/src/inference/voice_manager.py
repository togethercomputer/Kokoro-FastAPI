"""Voice management with controlled resource handling."""

import asyncio
from typing import Dict, List, Optional, Union

import torch

from ..core import paths
from ..core.config import settings


class VoiceManager:
    """Manages voice loading and caching with controlled resource usage."""

    # Singleton instance
    _instance: Optional['VoiceManager'] = None

    def __init__(self):
        """Initialize voice manager."""
        # Strictly respect settings.use_gpu
        self._device = settings.get_device()
        self._voices: Dict[str, torch.Tensor] = {}

    async def get_voice_path(self, voice_name: str) -> str:
        """Get path to voice file.

        Args:
            voice_name: Name of voice

        Returns:
            Path to voice file

        Raises:
            RuntimeError: If voice not found
        """
        return await paths.get_voice_path(voice_name)

    async def load_voice(
        self, voice_name: str, device: Optional[str] = None
    ) -> torch.Tensor:
        """Load voice tensor.

        Args:
            voice_name: Name of voice to load
            device: Optional override for target device

        Returns:
            Voice tensor

        Raises:
            RuntimeError: If voice not found
        """
        try:
            voice_path = await self.get_voice_path(voice_name)
            target_device = device or self._device
            voice = await paths.load_voice_tensor(voice_path, target_device)
            self._voices[voice_name] = voice
            return voice
        except Exception as e:
            raise RuntimeError(f"Failed to load voice {voice_name}: {e}")

    def load_voice_sync(
        self, voice_name: str, device: Optional[str] = None
    ) -> torch.Tensor:
        """Load voice tensor synchronously.
        
        This is a synchronous wrapper around the async load_voice method.
        Use this when you need to call from non-coroutine contexts.

        Args:
            voice_name: Name of voice to load
            device: Optional override for target device

        Returns:
            Voice tensor

        Raises:
            RuntimeError: If voice not found
        """
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're in an event loop, we can't use asyncio.run()
            # Instead, we need to create a new thread or use a different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.get_voice_path(voice_name))
                path = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            path = asyncio.run(self.get_voice_path(voice_name))
        return torch.load(path, map_location=device)

    async def combine_voices(
        self, voices: List[str], device: Optional[str] = None
    ) -> torch.Tensor:
        """Combine multiple voices.

        Args:
            voices: List of voice names to combine
            device: Optional override for target device

        Returns:
            Combined voice tensor

        Raises:
            RuntimeError: If any voice not found
        """
        if len(voices) < 2:
            raise ValueError("Need at least 2 voices to combine")

        target_device = device or self._device
        voice_tensors = []
        for name in voices:
            voice = await self.load_voice(name, target_device)
            voice_tensors.append(voice)

        combined = torch.mean(torch.stack(voice_tensors), dim=0)
        return combined

    def combine_voices_sync(
        self, voices: List[str], device: Optional[str] = None
    ) -> torch.Tensor:
        """Combine multiple voices synchronously.
        
        This is a synchronous wrapper around the async combine_voices method.

        Args:
            voices: List of voice names to combine
            device: Optional override for target device

        Returns:
            Combined voice tensor

        Raises:
            RuntimeError: If any voice not found
        """
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're in an event loop, we can't use asyncio.run()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.combine_voices(voices, device))
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.combine_voices(voices, device))

    async def list_voices(self) -> List[str]:
        """List available voice names.

        Returns:
            List of voice names
        """
        return await paths.list_voices()

    def list_voices_sync(self) -> List[str]:
        """List available voice names synchronously.
        
        This is a synchronous wrapper around the async list_voices method.

        Returns:
            List of voice names
        """
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're in an event loop, we can't use asyncio.run()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.list_voices())
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.list_voices())

    def cache_info(self) -> Dict[str, Union[int, str]]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        return {"loaded_voices": len(self._voices), "device": self._device}


async def get_manager() -> VoiceManager:
    """Get voice manager instance.

    Returns:
        VoiceManager instance
    """
    if VoiceManager._instance is None:
        VoiceManager._instance = VoiceManager()
    return VoiceManager._instance


def get_manager_sync() -> VoiceManager:
    """Get voice manager instance synchronously.

    Returns:
        VoiceManager instance
    """
    if VoiceManager._instance is None:
        VoiceManager._instance = VoiceManager()
    return VoiceManager._instance
