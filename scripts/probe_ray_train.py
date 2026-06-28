"""Probe the Ray Train APIs used by GraphIDS."""

from __future__ import annotations

import json

from graphids.exp.ray_backend import probe_ray_train_imports


def main() -> None:
    print(json.dumps(probe_ray_train_imports(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
