#!/usr/bin/env python3
"""Check OpenAPI spec for salary, travelExpense, dimension, creditNote endpoints."""
import json

with open('openapi.json') as f:
    api = json.load(f)

paths = api.get('paths', {})
components = api.get('components', {}).get('schemas', {})

def resolve_ref(ref, schemas):
    name = ref.split('/')[-1]
    return schemas.get(name, {}), name

def show_schema(schema, schemas, depth=0):
    if '$ref' in schema:
        resolved, name = resolve_ref(schema['$ref'], schemas)
        return show_schema(resolved, schemas, depth)
    required = schema.get('required', [])
    props = schema.get('properties', {})
    for pname, pval in props.items():
        req = " [REQUIRED]" if pname in required else ""
        ptype = pval.get('type', '')
        ref = pval.get('$ref', '')
        items = pval.get('items', {})
        if ref:
            _, rname = resolve_ref(ref, schemas)
            print(f"{'  '*(depth+1)}{pname}: {rname}{req}")
        elif items and items.get('$ref'):
            _, rname = resolve_ref(items['$ref'], schemas)
            print(f"{'  '*(depth+1)}{pname}: array of {rname}{req}")
        else:
            print(f"{'  '*(depth+1)}{pname}: {ptype}{req}")

# Salary transaction
print("=" * 60)
print("POST /salary/transaction")
print("=" * 60)
spec = paths.get('/salary/transaction', {}).get('post', {})
if spec:
    body = spec.get('requestBody', {}).get('content', {}).get('application/json', {}).get('schema', {})
    if body:
        show_schema(body, components)
    params = spec.get('parameters', [])
    for p in params:
        print(f"  param: {p['name']} required={p.get('required')} type={p.get('schema',{}).get('type','')}")

# SalaryTransaction, SalaryV2Transaction schemas
for name in sorted(components.keys()):
    if 'salary' in name.lower() or 'payslip' in name.lower():
        s = components[name]
        if s.get('properties'):
            print(f"\nSchema: {name}")
            print(f"  required: {s.get('required', [])}")
            for pname, pval in s.get('properties', {}).items():
                ref = pval.get('$ref', '')
                items = pval.get('items', {})
                ptype = pval.get('type', '')
                if ref:
                    rn = ref.split('/')[-1]
                    ptype = f"ref:{rn}"
                elif items and items.get('$ref'):
                    rn = items['$ref'].split('/')[-1]
                    ptype = f"array of {rn}"
                print(f"    {pname}: {ptype}")

# Voucher posting - check for customDimension vs freeAccountingDimension
print("\n" + "=" * 60)
print("Voucher Posting properties (looking for dimension fields)")
print("=" * 60)
posting_schema = components.get('Posting', {})
for pname in sorted(posting_schema.get('properties', {}).keys()):
    if 'dimension' in pname.lower() or 'free' in pname.lower() or 'custom' in pname.lower():
        pval = posting_schema['properties'][pname]
        ref = pval.get('$ref', '')
        print(f"  {pname}: {ref or pval.get('type', '?')}")

# PerDiemCompensation
print("\n" + "=" * 60)
print("POST /travelExpense/perDiemCompensation")
print("=" * 60)
spec = paths.get('/travelExpense/perDiemCompensation', {}).get('post', {})
if spec:
    body = spec.get('requestBody', {}).get('content', {}).get('application/json', {}).get('schema', {})
    if body:
        show_schema(body, components)

pdc = components.get('PerDiemCompensation', {})
print(f"\nPerDiemCompensation schema:")
print(f"  required: {pdc.get('required', [])}")
for pname, pval in pdc.get('properties', {}).items():
    ref = pval.get('$ref', '')
    ptype = pval.get('type', ref)
    print(f"    {pname}: {ptype}")

# TravelCost
print("\n" + "=" * 60)  
print("TravelCost schema")
print("=" * 60)
tc = components.get('TravelCost', {})
print(f"  required: {tc.get('required', [])}")
for pname, pval in tc.get('properties', {}).items():
    ref = pval.get('$ref', '')
    ptype = pval.get('type', ref)
    print(f"    {pname}: {ptype}")

# Credit note endpoint
print("\n" + "=" * 60)
print("PUT /invoice/{id}/:createCreditNote")  
print("=" * 60)
for path_key in paths:
    if 'creditNote' in path_key.lower() or 'credit' in path_key.lower():
        print(f"  {path_key}")
        for method, spec in paths[path_key].items():
            params = spec.get('parameters', [])
            for p in params:
                print(f"    param: {p['name']} required={p.get('required')} type={p.get('schema',{}).get('type','')}")
