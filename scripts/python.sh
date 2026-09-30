#!/bin/sh
# Project-local workaround for Homebrew Python loading Apple's older libexpat.
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -f /opt/homebrew/opt/expat/lib/libexpat.1.dylib ]; then
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
export PATH="$HOME/.lmstudio/bin:$PATH"
exec "$project_dir/.venv/bin/python" "$@"
