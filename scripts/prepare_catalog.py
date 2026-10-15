"""Stream the pinned ETO key-cleaning step before its in-memory JSON load."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from .common import DEFAULT_ETO, read_json, sha256, write_json


def cleaner_from_source(eto_root):
    path = eto_root / "envs/webshop/src/webshop/web_agent_site/engine/engine.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "clean_product_keys")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"print": lambda *args, **kwargs: None}
    exec(compile(module, str(path), "exec"), namespace)
    digest = hashlib.sha256(ast.dump(function).encode()).hexdigest()
    return namespace["clean_product_keys"], digest


def prepare_catalog(eto_root, source_hash=None):
    source = eto_root / "envs/webshop/data/items_shuffle.json"
    target = source.with_name("items_shuffle.cleaned.json")
    manifest_path = target.with_suffix(".manifest.json")
    cleaner, cleaner_hash = cleaner_from_source(eto_root)
    identity = {"source_sha256": source_hash or sha256(source), "cleaner_sha256": cleaner_hash,
                "schema_version": 1}
    if target.is_file() and manifest_path.is_file():
        metadata = read_json(manifest_path)
        if all(metadata.get(key) == value for key, value in identity.items()) and sha256(target) == metadata.get("cleaned_sha256"):
            return target, metadata

    import ijson
    temporary = target.with_suffix(".json.tmp")
    print("Streaming all products through ETO's existing key cleaner...", flush=True)
    count = 0
    with source.open("rb") as incoming, temporary.open("w", encoding="utf-8", newline="\n") as outgoing:
        outgoing.write("[")
        for product in ijson.items(incoming, "item", use_float=True):
            cleaned = cleaner([product])[0]
            if count:
                outgoing.write(",")
            outgoing.write(json.dumps(cleaned, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
            count += 1
            if count % 100000 == 0:
                print(f"Catalog cache: {count} products preserved", flush=True)
        outgoing.write("]\n")
    temporary.replace(target)
    metadata = {**identity, "product_count": count, "cleaned_sha256": sha256(target)}
    write_json(manifest_path, metadata)
    print(f"Catalog cache complete: {count} products, {target.stat().st_size / 1e9:.2f} GB", flush=True)
    return target, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eto-root", type=Path, default=DEFAULT_ETO)
    args = parser.parse_args()
    from .generate_teacher import verify_upstream
    verify_upstream(args.eto_root)
    prepare_catalog(args.eto_root)


if __name__ == "__main__":
    main()
