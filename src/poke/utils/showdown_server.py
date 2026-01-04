"""Utilities for managing local Showdown server."""
import subprocess
import time
import socket
from pathlib import Path

class ShowdownServer:
    def __init__(self, showdown_path: str = "./pokemon-showdown", port: int = 8000):
        self.showdown_path = Path(showdown_path)
        self.port = port
        self.process = None

    def start(self, timeout: int = 30) -> None:
        """Start the Showdown server."""
        if self.is_running():
            return

        self.process = subprocess.Popen(
            ["node", "pokemon-showdown", "start", "--no-security"],
            cwd=self.showdown_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._check_port():
                return
            time.sleep(0.5)
        raise TimeoutError(f"Server did not start within {timeout}s")

    def stop(self) -> None:
        """Stop the Showdown server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._check_port()

    def _check_port(self) -> bool:
        """Check if port is accepting connections."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(("localhost", self.port))
            return True
        except ConnectionRefusedError:
            return False
        finally:
            sock.close()
