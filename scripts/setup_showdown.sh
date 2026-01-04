#!/bin/bash
set -e

SHOWDOWN_DIR="./pokemon-showdown"

if [ ! -d "$SHOWDOWN_DIR" ]; then
    git clone https://github.com/smogon/pokemon-showdown.git "$SHOWDOWN_DIR"
fi

cd "$SHOWDOWN_DIR"
npm install
cp config/config-example.js config/config.js
