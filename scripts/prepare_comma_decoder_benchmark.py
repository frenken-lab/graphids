"""Download three independent commaCarSegments Forester routes and paired DBC.

Requires requests, pycapnp, and zstandard. Output CSVs use GraphIDS raw-CAN format.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import capnp
import requests
import zstandard

DATA_REV = "edb6480d0e0b5775fde7150c01bcdd5b78d0ad32"
PILOT_REV = "ec95db3f1fa19f76940497fedfdb62e09ea19912"
DBC_REV = "35f7e0813462607ef1d703e52313e7571e31405d"
HF = f"https://huggingface.co/datasets/commaai/commaCarSegments/resolve/{DATA_REV}"
PILOT = f"https://raw.githubusercontent.com/commaai/openpilot/{PILOT_REV}/openpilot/cereal"
DBC = f"https://raw.githubusercontent.com/commaai/opendbc/{DBC_REV}/opendbc/dbc/generator/subaru"
ROUTES = (
    ("2dc45626c10c8bba/00000000--a4e80ad2ff/5", "d5eddb82e0f518dc0a1bb15e75749e7793af92141161740ecbc182a5bcb06051"),
    ("49b23b64a6e11383/00000200--a9db57c0c5/5", "7c5f574acb381db44c9ce803706670218d2b391580bf4c8b9196c22b420b16ab"),
    ("9f50d1d2db783371/00000013--b22071797d/11", "397dbcbcb1397a4e94dc8c8235436d32ee4df98059bf08dcd627955c581e0a49"),
)


def fetch(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(fetch(f"{HF}/database.json"))
    if any(route + "/s" not in manifest["SUBARU_FORESTER"] for route, _ in ROUTES):
        raise RuntimeError("pinned route absent from Forester manifest")
    if len({route.split('/', 1)[0] for route, _ in ROUTES}) != 3:
        raise RuntimeError("routes must come from distinct devices")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "include").mkdir()
        for name in ("log.capnp", "custom.capnp", "deprecated.capnp"):
            (root / name).write_bytes(fetch(f"{PILOT}/{name}"))
        (root / "car.capnp").write_bytes(fetch(
            f"https://raw.githubusercontent.com/commaai/opendbc/{DBC_REV}/opendbc/car/car.capnp"))
        # The import is metadata for C++ code generation; pycapnp only needs a stub.
        (root / "include/c++.capnp").write_text(
            '@0xb4e3f8fa30b9318b; annotation namespace(file) :Text;\n')
        schema = capnp.load(str(root / "log.capnp"), imports=[str(root)])
        base = fetch(f"{DBC}/_subaru_global.dbc").decode()
        extension = fetch(f"{DBC}/subaru_global_2017.dbc").decode()
        (args.output / "subaru_global_2017.dbc").write_text(
            base + "\n" + extension.split("\n", 1)[1])
        records = []
        for number, (route, expected_hash) in enumerate(ROUTES):
            url = f"{HF}/segments/{route}/rlog.zst"
            compressed = fetch(url)
            digest = hashlib.sha256(compressed).hexdigest()
            if digest != expected_hash:
                raise RuntimeError(f"compressed segment hash changed: {route}")
            raw = root / "rlog"
            with raw.open("wb") as handle:
                handle.write(zstandard.ZstdDecompressor().decompress(
                    compressed, max_output_size=200_000_000))
            path = args.output / f"reference_{number + 1}.csv"
            count = 0
            with raw.open("rb") as handle, path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(("timestamp", "arbitration_id", "data_field", "attack"))
                for event in schema.Event.read_multiple(handle):
                    if event.which() != "can":
                        continue
                    for frame in event.can:
                        if frame.src == 0:
                            writer.writerow((event.logMonoTime / 1e9,
                                             format(frame.address, "X"),
                                             bytes(frame.dat).hex(), 0))
                            count += 1
            records.append({"route": route, "url": url,
                            "compressed_sha256": digest,
                            "dataset_revision": DATA_REV, "opendbc_revision": DBC_REV,
                            "frames_bus_0": count, "csv": str(path)})
    (args.output / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
