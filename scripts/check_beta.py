#!/usr/bin/env python3
"""Cross-reference our API calls against [BETA] endpoints in openapi.json."""
import json, re

with open('openapi.json') as f:
    api = json.load(f)

paths = api.get('paths', {})
beta_set = set()
for path, methods in paths.items():
    for method, spec in methods.items():
        if method in ('get', 'post', 'put', 'delete', 'patch'):
            summary = spec.get('summary', '')
            desc = spec.get('description', '')
            if '[BETA]' in summary or '[BETA]' in desc or '[Beta]' in summary:
                beta_set.add((method.upper(), path))

with open('agent/executor.py') as f:
    code = f.read()

method_map = {
    'get': 'GET', 'get_list': 'GET',
    'post': 'POST', 'post_value': 'POST',
    'put': 'PUT', 'delete': 'DELETE',
}

used_endpoints = []
for line_no, line in enumerate(code.split('\n'), 1):
    m = re.search(r'client\.(get|get_list|post|post_value|put|delete)\s*\(', line)
    if not m:
        continue
    method_raw = m.group(1)
    method = method_map.get(method_raw, method_raw.upper())
    # Extract path from the rest of the line
    rest = line[m.end():]
    pm = re.search(r'[f]?["\'](/[^"\']+)["\']', rest)
    if not pm:
        continue
    path = pm.group(1)
    path = re.sub(r'\?.*', '', path)  # strip query string
    path = re.sub(r'\{[^}]+\}', '{id}', path)  # normalize interpolation
    used_endpoints.append((method, path, line_no))

# Check against BETA
print("=== ENDPOINTS WE USE THAT ARE [BETA] ===")
beta_found = []
for method, path, line_no in sorted(used_endpoints, key=lambda x: x[2]):
    for bm, bp in beta_set:
        if bm != method:
            continue
        bp_norm = re.sub(r'\{[^}]+\}', '{id}', bp)
        if bp_norm == path:
            beta_found.append((line_no, method, path, bp))
            break

if beta_found:
    for ln, m, p, bp in beta_found:
        print(f"  L{ln}: {m} {p}  -->  BETA spec: {bp}")
else:
    print("  None found!")

print()
print("=== ALL UNIQUE ENDPOINTS WE USE ===")
unique = sorted(set((m, p) for m, p, _ in used_endpoints))
for method, path in unique:
    is_beta = ""
    for bm, bp in beta_set:
        bp_norm = re.sub(r'\{[^}]+\}', '{id}', bp)
        if bm == method and bp_norm == path:
            is_beta = " *** BETA ***"
            break
    print(f"  {method} {path}{is_beta}")
