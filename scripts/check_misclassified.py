#!/usr/bin/env python3
"""Check specific misclassified episodes."""
import json

eps = [json.loads(l) for l in open("logs/episodes.jsonl") if l.strip()]

targets = [
    "une commande",
    "Nous avons envoy",
    "Registre 11 horas",
    "Enregistrez 16 heure",
    "Opprett en fri regns",
    "Gehaltsabrechnung",
    "bankutskrifta",
    "Crea tres departamentos",
    "Erstellen Sie drei Abteilungen",
    "Betalinga fr",
    "delete_voucher",
]

seen = set()
for ep in eps:
    p = ep.get("prompt", "")
    for t in targets:
        if t in p or t in ep.get("task_type", ""):
            key = p[:60]
            if key in seen:
                continue
            seen.add(key)
            rev = (ep.get("revision", "") or ep.get("git_sha", ""))[-8:]
            task = ep.get("task_type", "?")
            calls = len(ep.get("api_calls", []))
            out = ep.get("outcome", "?")
            errs = len([c for c in ep.get("api_calls", []) if c.get("status", 200) >= 400])
            print(f"[{rev}] {task:30s} calls={calls} 4xx={errs} out={out}")
            print(f"  {p[:250]}")
            entities = ep.get("parsed_entities", {})
            for k, v in entities.items():
                if k == "_raw_prompt":
                    continue
                print(f"  ENTITY.{k}: {json.dumps(v, ensure_ascii=False)[:200]}")
            # Show API calls
            for c in ep.get("api_calls", []):
                s = c.get("status", 0)
                if s >= 400:
                    print(f"  API: {c.get('method','?')} {c.get('path','?')} -> {s}")
            print()
