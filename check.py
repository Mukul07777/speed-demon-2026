#!/usr/bin/env python3
"""Build one track the way the Speed Demon board does, run it, and check its output.

    python3 check.py a --data data/a [--expected <sha256>] [--skip-build]

The build is offline (--network=none), like the board's. The run uses the board's
sandbox settings: no network, a read-only filesystem, an unprivileged user, 8 GB of
memory, 512 processes and 1 GB of /tmp. The time printed includes container start-up;
the board times your process alone, on its own machine.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

OUTPUT = {"a": "summary.txt", "b": "results.txt", "c": "aggregates.txt", "d": "distances.txt"}
ALLOWLIST_URL = "https://speeddemon.dev/allowlist.json"
IMAGE_MAX_BYTES = 2 * 1024 ** 3
ROOT = os.path.dirname(os.path.abspath(__file__))


def fail(message):
    print(f"FAIL  {message}")
    sys.exit(1)


def allowlisted_digests():
    # The site's CDN refuses urllib's default User-Agent, so name the client.
    req = urllib.request.Request(ALLOWLIST_URL, headers={"User-Agent": "speed-demon-starter-check/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {e["digest"] for e in json.load(resp)["images"]}
    except Exception as exc:  # offline or unreachable: the board still checks
        print(f"note  could not fetch {ALLOWLIST_URL} ({type(exc).__name__}); skipping the FROM check")
        return None


def from_problems(dockerfile, digests):
    """FROM lines the board would refuse: not an earlier stage, not pinned, or not allowlisted."""
    stages, problems = set(), []
    for raw in dockerfile.splitlines():
        words = [w for w in raw.split() if not w.startswith("--")]
        if len(words) < 2 or words[0].upper() != "FROM":
            continue
        image = words[1]
        if image.lower() not in stages:
            if "@sha256:" not in image:
                problems.append(f"FROM {image} is not pinned by digest")
            elif image.split("@", 1)[1] not in digests:
                problems.append(f"FROM {image} is not in the allowlist")
        if len(words) >= 4 and words[2].upper() == "AS":
            stages.add(words[3].lower())
    return problems


def build(track, tag):
    print(f"build docker build --network=none -t {tag} track-{track}")
    if subprocess.run(["docker", "build", "--network=none", "-t", tag, os.path.join(ROOT, f"track-{track}")]).returncode:
        fail("the build failed; it has to work offline, so commit your dependencies")
    size = int(subprocess.run(["docker", "image", "inspect", "-f", "{{.Size}}", tag],
                              capture_output=True, text=True, check=True).stdout.strip())
    if size > IMAGE_MAX_BYTES:
        fail(f"the image is {size / 1e9:.2f} GB; the board's cap is 2 GB")
    print(f"ok    image is {size / 1e6:.0f} MB")


def run(track, tag, data):
    out = os.path.join(ROOT, ".out", f"track-{track}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    os.chmod(out, 0o777)  # the run is an unprivileged user, as on the board
    cmd = ["docker", "run", "--rm", "--network=none", "--read-only", "--tmpfs", "/tmp:rw,size=1g,exec",
           "--memory=8g", "--memory-swap=8g", "--pids-limit=512", "--cap-drop=ALL",
           "--security-opt=no-new-privileges", "--user", "65534:65534", "--ulimit", "nofile=8192:8192",
           "-e", f"TRACK={track}", "-v", f"{data}:/data:ro", "-v", f"{out}:/out:rw", tag]
    started = time.monotonic()
    code = subprocess.run(cmd).returncode
    took = time.monotonic() - started
    if code:
        fail(f"the run exited {code} after {took:.2f} s")
    return out, took


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Build and run one Speed Demon track the way the board does.")
    ap.add_argument("track", choices=sorted(OUTPUT))
    ap.add_argument("--data", required=True, help="the folder holding the track's input files")
    ap.add_argument("--expected", help="the sha256 the output must match")
    ap.add_argument("--skip-build", action="store_true", help="run the image from the last build")
    args = ap.parse_args()

    if shutil.which("docker") is None:
        fail("docker is not on PATH")
    dockerfile = os.path.join(ROOT, f"track-{args.track}", "Dockerfile")
    if not os.path.isfile(dockerfile):
        fail(f"track-{args.track}/Dockerfile is missing")
    data = os.path.abspath(args.data)
    if not os.path.isdir(data):
        fail(f"{data} is not a folder")

    tag = f"speeddemon-check/track-{args.track}"
    if not args.skip_build:
        digests = allowlisted_digests()
        if digests is not None:
            with open(dockerfile, encoding="utf-8") as f:
                problems = from_problems(f.read(), digests)
            if problems:
                fail("; ".join(problems) + " (the board refuses this)")
            print("ok    every FROM is allowlisted")
        build(args.track, tag)

    out, took = run(args.track, tag, data)
    name = OUTPUT[args.track]
    found = sorted(os.listdir(out))
    if found != [name]:
        fail(f"/out must hold exactly {name}, found {found}")
    digest = sha256(os.path.join(out, name))
    print(f"ok    {name} sha256 {digest}")
    print(f"time  {took:.2f} s, including container start-up; the board's number will differ")
    if args.expected:
        if digest != args.expected.strip().lower():
            fail(f"the output does not match the expected sha256 {args.expected}")
        print("PASS  the output matches the expected sha256")
    else:
        print("note  no --expected given. If this folder is unchanged, that sha256 is the right answer; check against it:")
        print(f"      python3 check.py {args.track} --data {args.data} --expected {digest}")


if __name__ == "__main__":
    main()
