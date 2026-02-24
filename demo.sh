#!/usr/bin/env bash
# Octane 3-turn demo — recorded with asciinema
# Run: asciinema rec demo.cast --command ./demo.sh

cd "$(dirname "$0")"

printf 'what is NVDA stock price today?\nwrite a python script to print the number 189.82\nwhat did I research earlier in this session?\nexit\n' \
  | OCTANE_LOG_LEVEL=WARNING sandbox/oct_env/bin/python -m octane.main chat 2>/dev/null
