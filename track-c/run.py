"""Container entrypoint for the reference implementations.

The harness runs every image with `-e TRACK=<a|b|c|d>`, so one reference
image serves all four tracks.
"""
import importlib
import os
import sys

MODULES = {"a": "a_ingest", "b": "b_index", "c": "c_join", "d": "d_traverse"}


def run():
    track = os.environ.get("TRACK", "").strip().lower()
    if track not in MODULES:
        sys.stderr.write(f"TRACK must be one of {sorted(MODULES)}, got {track!r}\n")
        return 2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    importlib.import_module(MODULES[track]).main("/data", "/out")
    return 0


if __name__ == "__main__":
    sys.exit(run())
