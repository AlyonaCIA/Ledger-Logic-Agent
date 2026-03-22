#!/usr/bin/env python3
"""Cross-reference our API calls against [BETA] endpoints in openapi.json."""
import json, re

with open('openapi.json') as f:
    api = json.load(f)

paths_spec = api.get('paths', {})

spec_lookup = {}
beta_set = set()
for path, methods in paths_spec.items():
    for method, spec in methods.items():
        if method in ('get', 'post', 'put', 'delete', 'patch'):
            summary = spec.get('summary', '')
            is_beta = '[BETA]' in summary or '[Beta]' in summary
            path_norm = re.sub(r'\{[^}]+\}', '{id}', path)
            key = (method.upper(), path_norm)
            spec_lookup[key] = (summary[:80], is_beta, path)
            if is_beta:
                beta_set.add(key)

with open('agent/executor.py') as f:
    code = f.read()

method_map = {
    'get': 'GET', 'get_list': 'GET', 'get_value': 'GET',
    'post': 'POST', 'post_value': 'POST',
    'put': 'PUT', 'put_value': 'PUT',
    'delete': 'DELETE',
}

used_endpoints = []
for line_no, line in enumerate(code.split('\n'), 1):
    m = re.search(r'client\.(get_list|get_value|get|post_value|post|put_value|put|delete)\s*\(', line)
    if not m:
        continue
    method_raw = m.group(1)
    method = method_map.get(method_raw, method_raw.upper())
    rest = line[m.end():]
    pm = re.search(r'f?["\'](/[^"\']+)["\']', rest)
    if not pm:
        continue
    path = pm.group(1)
    path = re.sub(r'\?.*', '', path)
    path = re.sub(r'\{[^}]*\}', '{id}', path)
    used_endpoints.append((method, path, line_no))

print("=== ENDPOINTS WE USE THAT ARE [BETA] ===")
beta_found = []
for method, path, line_no in sorted(used_endpoints, key=lambda x: x[2]):
    key = (method, path)
    if key in beta_set:
        summary, _, orig = spec_lookup[key]
        beta_found.append(f"  L{line_no}: {method} {path}  -->  {summary}")

if beta_found:
    for line in beta_found:
        print(line)
else:
    print("  None found!")

print()
print("=== ENDPOINTS NOT IN SPEC (could be blocked) ===")
not_in_spec = []
for method, path, line_no in sorted(used_endpoints, key=lambda x: x[2]):
    key = (method, path)
    if key not in spec_lookup:
        not_in_spec.append(f"  L{line_no}: {method} {path}")
if not_in_spec:
    for line in sorted(set(not_in_spec)):
        print(line)
else:
    print("  All endpoints found in spec!")

print()
print("=== ALL UNIQUE ENDPOINTS WE USE ===")
unique = sorted(set((m, p) for m, p, _ in used_endpoints))
for method, path in unique:
    key = (method, path)
    if key in beta_set:
        summary, _, _ = spec_lookup[key]
        print(f"  {method} {path} *** BETA *** [{summary}]")
    elif key not in spec_lookup:
        print(f"  {method} {path} *** NOT IN SPEC ***")
    else:
        print(f"  {method} {path}")
