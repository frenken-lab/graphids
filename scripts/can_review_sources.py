"""Download commit-pinned original logs and label files for an explicit review plan.

Run in a CPU Slurm allocation. Downloads alone never approve a recording manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

COMMITS = {
    "can-dataset": "c7e146cc75a5bac5d79e2a064e300e2276d24c51",
    "can-ml": "7268a6f564b7a1d4ad2bec089b60abaf68e18bdf",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-index", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prior = json.loads(args.reuse_index.read_text()) if args.reuse_index else []
    reused = {item["url"]: item for item in prior if "error" not in item}
    plan = json.loads(args.plan.read_text())
    index = []
    for local, item in plan["assignments"].items():
        for field, repo in (("source_log", "can-dataset"), ("source_labels", "can-ml")):
            if field not in item:
                continue
            source = item[field]
            repo = item.get(f"{field}_repo", repo)
            commit = COMMITS[repo]
            quoted = urllib.parse.quote(source, safe="/")
            url = f"https://api.bitbucket.org/2.0/repositories/brooke-lampe/{repo}/src/{commit}/{quoted}"
            destination = args.output / source.replace("/", "-")
            entry = dict(
                local=local, source=source, repository=repo, commit=commit, url=url
            )
            previous = reused.get(url)
            if previous and Path(previous["download"]).is_file():
                destination = Path(previous["download"])
            elif not destination.exists():
                errors = []
                alternatives = (
                    url,
                    f"https://bitbucket.org/brooke-lampe/{repo}/raw/{commit}/{quoted}",
                )
                for candidate in alternatives:
                    temporary = destination.with_suffix(destination.suffix + ".partial")
                    try:
                        request = urllib.request.Request(
                            candidate,
                            headers={"User-Agent": "GraphIDS provenance review"},
                        )
                        with (
                            urllib.request.urlopen(request, timeout=120) as response,
                            temporary.open("wb") as handle,
                        ):
                            if response.headers.get_content_type() in (
                                "text/html",
                                "application/json",
                            ):
                                raise ValueError(
                                    "expected raw recording, received HTML/JSON"
                                )
                            shutil.copyfileobj(response, handle)
                        if not temporary.stat().st_size:
                            raise ValueError("empty upstream recording")
                        temporary.replace(destination)
                        entry["download_url"] = candidate
                        break
                    except Exception as exc:
                        temporary.unlink(missing_ok=True)
                        errors.append(f"{candidate}: {exc}")
                else:
                    raise RuntimeError("; ".join(errors))
            with destination.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            entry.update(
                download=str(destination.resolve()),
                bytes=destination.stat().st_size,
                sha256=digest,
            )
            index.append(entry)
            (args.output / "sources.json").write_text(
                json.dumps(index, indent=2) + "\n"
            )
            print(f"available {source}: {entry['bytes']} bytes", flush=True)


if __name__ == "__main__":
    main()
