#!/usr/bin/env python3
"""Check API spec details for specific endpoints."""
import json

with open('openapi.json') as f:
    api = json.load(f)

paths = api.get('paths', {})

# 1. Travel expense deliver/approve
for ep in ['/travelExpense/:deliver', '/travelExpense/:approve']:
    spec = paths.get(ep, {}).get('put', {})
    print(f'=== PUT {ep} ===')
    params = spec.get('parameters', [])
    for p in params:
        print(f'  param: {p.get("name")} in={p.get("in")} required={p.get("required")}')
    body = spec.get('requestBody', {})
    if body:
        print(f'  requestBody: {json.dumps(body)[:300]}')
    print()

# 2. supplierInvoice addPayment
for ep in ['/supplierInvoice/{invoiceId}/:addPayment']:
    for method in ['post', 'put']:
        spec = paths.get(ep, {}).get(method, {})
        if spec:
            print(f'=== {method.upper()} {ep} ===')
            params = spec.get('parameters', [])
            for p in params:
                print(f'  param: {p.get("name")} in={p.get("in")} required={p.get("required")}')
            print()

# 3. Company endpoints (all)
print('=== COMPANY ENDPOINTS ===')
for p in sorted(paths.keys()):
    if p.startswith('/company'):
        for method in ['get', 'put', 'post', 'delete']:
            if method in paths[p]:
                s = paths[p][method].get('summary', '')
                beta = ' [BETA]' if '[BETA]' in s else ''
                print(f'  {method.upper()} {p}{beta} -- {s[:60]}')
