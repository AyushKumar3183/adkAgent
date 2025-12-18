"""Session memory for tracking generated and edited images."""
from typing import Optional


class SessionMemory:
    """In-memory storage for session state."""
    
    def __init__(self):
        self._last_generated: Optional[str] = None
        self._last_edited: Optional[str] = None
        self._recent_images: list[str] = []
    
    def set_last_generated(self, filename: str):
        """Set the last generated image filename."""
        self._last_generated = filename
        if filename not in self._recent_images:
            self._recent_images.insert(0, filename)
            # Keep only last 10 images
            self._recent_images = self._recent_images[:10]
    
    def set_last_edited(self, filename: str):
        """Set the last edited image filename."""
        self._last_edited = filename
        if filename not in self._recent_images:
            self._recent_images.insert(0, filename)
            # Keep only last 10 images
            self._recent_images = self._recent_images[:10]
    
    def get_last_generated(self) -> Optional[str]:
        """Get the last generated image filename."""
        return self._last_generated
    
    def get_last_edited(self) -> Optional[str]:
        """Get the last edited image filename."""
        return self._last_edited
    
    def get_most_recent(self) -> Optional[str]:
        """Get the most recent image (generated or edited)."""
        if self._recent_images:
            return self._recent_images[0]
        return self._last_generated or self._last_edited


# Global session memory instance
_memory: Optional[SessionMemory] = None


def get_memory() -> SessionMemory:
    """Get the global session memory instance."""
    global _memory
    if _memory is None:
        _memory = SessionMemory()
    return _memory
