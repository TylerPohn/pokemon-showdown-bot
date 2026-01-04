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
