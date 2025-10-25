#!/usr/bin/env bash
set -e

# Example: install evals and run a simple local review flow
pip install -r requirements.txt
pip install git+https://github.com/openai/evals.git@main

# Run the built-in benchmark runner to produce results.json
python benchmark/runner.py --vectorstore chroma --tests tests/memory_tests.json --output results.json

# At this point you can adapt results.json for openai-evals or run a custom eval script.
# For example, you could implement an evals "grader" that reads results.json and asks humans
# to score hallucination/consistency.

echo "Benchmark finished. results.json generated. See README for next steps."