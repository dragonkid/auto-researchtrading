#!/usr/bin/env python3
"""Resolve a Hermes custom_provider's credentials by name.

Reads the active Hermes profile config (default: quant-trading) and prints
    base_url<TAB>api_key<TAB>model
for the requested provider name. Used by autoresearch.sh to inject
ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL without ever
storing credentials in the (git-tracked) shell script.

Pure stdlib — the system python has no pyyaml, so we parse the regular
`custom_providers:` list block by hand. That block is flat and predictable:

    custom_providers:
    - name: skyapi
      base_url: https://api.skyapi.org/v1
      api_key: sk-...
      model: claude-opus-4-8

Usage:
    hermes_provider.py <provider_name> [--config PATH]
Exit codes:
    0  success (prints base_url\tapi_key\tmodel)
    1  provider not found / block missing (lists available names on stderr)
    2  config file unreadable
"""
import os
import re
import sys

DEFAULT_CONFIG = os.path.expanduser(
    "~/.hermes/profiles/quant-trading/config.yaml"
)


def parse_custom_providers(path):
    with open(path) as f:
        lines = f.readlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.rstrip() == "custom_providers:":
            start = i + 1
            break
    if start is None:
        return None
    # Each list item is a mapping whose keys may appear in ANY order (the
    # config serializer sorts them alphabetically, so "name" is usually LAST,
    # not first). So we collect every key of the current item into a dict and
    # only key it by its "name" field once the item is complete.
    items, cur = [], None
    for ln in lines[start:]:
        # The block ends at the next top-level mapping key (e.g. "image_gen:").
        # List items start with "-" which is also non-space, so we must NOT
        # treat a leading "-" as the block terminator.
        if re.match(r"^\S", ln) and not ln.lstrip().startswith("-"):
            break
        # New list item: "- key: value" (the dash may precede any key).
        m = re.match(r"^-\s+(\w+):\s*(.+?)\s*$", ln)
        if m:
            cur = {}
            items.append(cur)
            cur[m.group(1)] = m.group(2)
            continue
        # Continuation line of the current item: "  key: value".
        if cur is not None:
            m2 = re.match(r"^\s+(\w+):\s*(.+?)\s*$", ln)
            if m2:
                cur[m2.group(1)] = m2.group(2)
    provs = {}
    for it in items:
        nm = it.get("name")
        if nm:
            provs[nm] = it
    return provs


def main(argv):
    if len(argv) < 2:
        sys.stderr.write("usage: hermes_provider.py <name> [--config PATH]\n")
        return 2
    name = argv[1]
    config = DEFAULT_CONFIG
    if "--config" in argv:
        config = argv[argv.index("--config") + 1]
    if not os.path.isfile(config):
        sys.stderr.write(f"config not found: {config}\n")
        return 2
    provs = parse_custom_providers(config)
    if provs is None:
        sys.stderr.write(f"no custom_providers: block in {config}\n")
        return 1
    if name not in provs:
        sys.stderr.write(
            f"provider '{name}' not found. available: {', '.join(provs) or '(none)'}\n"
        )
        return 1
    p = provs[name]
    sys.stdout.write(
        f"{p.get('base_url','')}\t{p.get('api_key','')}\t{p.get('model','')}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
