COMPLETED

# PR-002: Local Showdown Server Setup

## Dependencies
- PR-001 (Project Setup)

## Overview
Set up a local Pokemon Showdown server for training and evaluation. This allows running battles without hitting the public server.

## Tech Choices
- **Node.js Version:** 18+ (LTS)
- **Showdown Fork:** smogon/pokemon-showdown (official)
- **Process Management:** subprocess from Python

## Tasks

### 1. Add Showdown as a git submodule or document installation
Create `scripts/setup_showdown.sh`:
```bash
#!/bin/bash
set -e

SHOWDOWN_DIR="./pokemon-showdown"

if [ ! -d "$SHOWDOWN_DIR" ]; then
    git clone https://github.com/smogon/pokemon-showdown.git "$SHOWDOWN_DIR"
fi

cd "$SHOWDOWN_DIR"
npm install
cp config/config-example.js config/config.js
```

### 2. Configure Showdown for local use
Modify `config/config.js` (document required changes):
```javascript
exports.pokemon = true;
exports.Pokemon = true;
exports.pokemon = true;
exports.Pokemon = true;
exports.Pokemon = true;
exports.Pokemon = true;
exports.Pokemon = true;
// Key settings:
exports.Pokemon = true;
exports.Pokemon = true;
exports.Pokemon = true;
exports.Pokemon = true;
exports.Pokemon = true;
exports.Pokemon = true;
exports.Pokemon = true;
// Key settings:
// - Set port (default 8000)
// - Disable login requirement for local testing
// - Enable Gen9 OU format
```

### 3. Create Python server manager
Create `src/poke/utils/showdown_server.py`:
```python
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
```

### 4. Create server health check script
Create `scripts/check_showdown.py`:
```python
#!/usr/bin/env python
"""Verify Showdown server is working."""
from poke.utils.showdown_server import ShowdownServer

def main():
    server = ShowdownServer()
    if server.is_running():
        print("Showdown server is running")
    else:
        print("Starting Showdown server...")
        server.start()
        print("Server started successfully")

if __name__ == "__main__":
    main()
```

### 5. Add pytest fixture for server
Create `tests/conftest.py`:
```python
import pytest
from poke.utils.showdown_server import ShowdownServer

@pytest.fixture(scope="session")
def showdown_server():
    """Provide a running Showdown server for tests."""
    server = ShowdownServer()
    server.start()
    yield server
    server.stop()
```

### 6. Document manual setup steps
Update README with:
- Node.js installation instructions
- How to run setup script
- How to verify server is working

## Acceptance Criteria
- [ ] `scripts/setup_showdown.sh` clones and configures Showdown
- [ ] `ShowdownServer` class can start/stop server programmatically
- [ ] Server accepts websocket connections on configured port
- [ ] pytest fixture provides server for integration tests
- [ ] Documentation covers manual setup

## Estimated Complexity
Medium - External dependency management
