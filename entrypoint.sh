#!/bin/bash
set -e

# Start llamafiler
echo "Starting llamafiler..."
/usr/local/bin/llamafiler -m all-MiniLM-L6-v2.F16.gguf -l 0.0.0.0:8080 -H "Access-Control-Allow-Origin: *" --trust 127.0.0.1/32 2> /tmp/llamafiler.logs &

# show llamafile start messages
sleep 1
head /tmp/llamafiler.logs

# Start marimo
cd byota/src/byota && marimo run --headless --host 0.0.0.0 --token --token-password byota notebook.py
