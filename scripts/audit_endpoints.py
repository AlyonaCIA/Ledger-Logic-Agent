#!/usr/bin/env python3
"""Audit executor.py API calls against openapi.json required fields."""
import json
import sys

with open('openapi.json') as f:
    api = json.load(f)

paths = api.get('paths', {})
components = api.get('components', {}).get('schemas', {})


def resolve_ref(ref, schemas):
    name = ref.split('/')[-1]
    return schemas.get(name, {})


def get_schema(schema, schemas):
    if '$ref' in schema:
        schema = resolve_ref(schema['$ref'], schemas)
    return schema


endpoints_to_check = [
    ('post', '/travelExpense'),
    ('post', '/travelExpense/cost'),
    ('post', '/travelExpense/perDiemCompensation'),
    ('post', '/employee/employment/details'),
    ('post', '/ledger/voucher'),
    ('post', '/salary/transaction'),
    ('post', '/supplier'),
    ('post', '/employee'),
    ('post', '/customer'),
    ('post', '/employee/employment'),
    ('put', '/invoice/{id}/:payment'),
    ('put', '/invoice/{id}/:createCreditNote'),
    ('post', '/timesheet/entry'),
    ('put', '/ledger/voucher/{id}/:sendToLedger'),
]

for method, path in endpoints_to_check:
    spec = paths.get(path, {}).get(method, {})
    if not spec:
        print(f'\n{method.upper()} {path}: NOT FOUND IN OPENAPI')
        continue
    body = spec.get('requestBody', {})
    schema = body.get('content', {}).get('application/json', {}).get('schema', {})
    schema = get_schema(schema, components)
    required = schema.get('required', [])
    props = list(schema.get('properties', {}).keys())
    
    # Also look at parameters (query params)
    params = spec.get('parameters', [])
    required_params = [p['name'] for p in params if p.get('required')]
    
    print(f'\n{method.upper()} {path}:')
    if required:
        print(f'  body required: {required}')
    if props:
        print(f'  body props: {props[:25]}')
    if required_params:
        print(f'  required query params: {required_params}')
    if params:
        param_names = [p['name'] for p in params]
        print(f'  all query params: {param_names[:15]}')

# Also check voucher posting fields
print('\n\n=== Voucher Posting fields ===')
voucher_schema = components.get('Voucher', {})
postings_prop = voucher_schema.get('properties', {}).get('postings', {})
if postings_prop and '$ref' in str(postings_prop):
    items = postings_prop.get('items', {})
    posting_schema = get_schema(items, components)
    required = posting_schema.get('required', [])
    props = list(posting_schema.get('properties', {}).keys())
    print(f'Posting required: {required}')
    print(f'Posting props: {props}')

# Check TravelCost fields
print('\n\n=== TravelCost fields ===')
tc = components.get('TravelCost', {})
required = tc.get('required', [])
props = list(tc.get('properties', {}).keys())
print(f'TravelCost required: {required}')
print(f'TravelCost props: {props}')

# Check PerDiemCompensation fields
print('\n\n=== PerDiemCompensation fields ===')
pdc = components.get('PerDiemCompensation', {})
required = pdc.get('required', [])
props = list(pdc.get('properties', {}).keys())
print(f'PerDiemCompensation required: {required}')
print(f'PerDiemCompensation props: {props}')

# Check Employment fields
print('\n\n=== Employment fields ===')
emp = components.get('Employment', {})
required = emp.get('required', [])
props = list(emp.get('properties', {}).keys())
print(f'Employment required: {required}')
print(f'Employment props: {props}')

# Check EmploymentDetails fields
print('\n\n=== EmploymentDetails fields ===')
ed = components.get('EmploymentDetails', {})
required = ed.get('required', [])
props = list(ed.get('properties', {}).keys())
print(f'EmploymentDetails required: {required}')
print(f'EmploymentDetails props: {props}')

# Check TravelExpense fields
print('\n\n=== TravelExpense fields ===')
te = components.get('TravelExpense', {})
required = te.get('required', [])
props = list(te.get('properties', {}).keys())
print(f'TravelExpense required: {required}')
print(f'TravelExpense props: {props}')
