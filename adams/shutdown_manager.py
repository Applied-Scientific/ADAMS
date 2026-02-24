"""
Graceful shutdown manager for handling Ctrl+C (SIGINT) and other termination signals.
Ensures cleanup of resources and proper session metadata updates.
"""

import signal
import sys
from typing import Callable, Optional


class ShutdownManager:
    """
    Manages graceful shutdown of the ADAMS application.
    
    Handles SIGINT (Ctrl+C) and SIGTERM signals to ensure:
    - Session metadata is saved
    - Trace files are properly closed
    - Resources are cleaned up
    """
    
    def __init__(self):
        self.shutdown_callbacks: list[Callable] = []
        self.shutdown_in_progress = False
        self._original_sigint = None
        self._original_sigterm = None
        
    def register_cleanup(self, callback: Callable) -> None:
        """
        Register a cleanup callback to be called on shutdown.
        
        Args:
            callback: Function to call during shutdown. Should not take arguments.
        """
        self.shutdown_callbacks.append(callback)
        
    def setup_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        
    def restore_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
            
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle termination signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        if self.shutdown_in_progress:
            print("\n[Force exit]", flush=True)
            sys.exit(1)
            
        self.shutdown_in_progress = True
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n Shutting down gracefully...]", flush=True)
        
        # Run all cleanup callbacks
        for callback in self.shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Warning: Cleanup callback failed: {e}", file=sys.stderr)
                
        # Exit cleanly
        sys.exit(0)
        
    def __enter__(self):
        """Context manager entry."""
        self.setup_handlers()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.restore_handlers()
        return False  # Don't suppress exceptions
