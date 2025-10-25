#!/usr/bin/env python3
"""
Runner: load tests, run benchmark, write results.json
Usage:
  python benchmark/runner.py --vectorstore chroma --tests tests/memory_tests.json --output results.json
"""
import argparse
import json
from benchmark.memory_benchmark import MemoryBenchmark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorstore", choices=["chroma","faiss"], default="chroma")
    parser.add_argument("--tests", default="tests/memory_tests.json")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    with open(args.tests, "r", encoding="utf-8") as f:
        tests = json.load(f)

    bench = MemoryBenchmark(vectorstore=args.vectorstore)
    results = bench.run_tests(tests)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()