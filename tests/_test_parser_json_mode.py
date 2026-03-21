"""Quick test: verify JSON mode fix eliminates parse_error for complex prompts."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.parser import _parse_task_sync

prompts = [
    (
        "NO multi-product order+payment",
        "Opprett en ordre for kunden Havbris AS (org.nr 913424921) med produktene Skylagring (1197) til 27100 kr og Programvarelisens (7613) til 4750 kr. Konverter ordren til faktura og registrer full betaling.",
        "create_invoice",
        "Havbris AS",
    ),
    (
        "FR multi-product commande",
        "Créez une commande pour le client Forêt SARL (nº org. 962176127) avec les produits Maintenance (2417) à 32250 NOK et Développement système (7053) à 16900 NOK. Convertissez la commande en facture.",
        "create_invoice",
        "Forêt SARL",
    ),
    (
        "DE employee creation",
        "Erstellen Sie einen neuen Mitarbeiter: Hans Schmidt (hans.schmidt@example.de), Telefon +4799887766.",
        "create_employee",
        None,
    ),
    (
        "PT customer creation",
        "Crie um cliente chamado Porto Digital Lda (e-mail: geral@portodigital.pt, org. nº 512345678).",
        "create_customer",
        "Porto Digital Lda",
    ),
]

passed = 0
failed = 0
for label, prompt, expected_task, expected_customer in prompts:
    result = _parse_task_sync(prompt, [])
    task = result.get("task_type")
    conf = result.get("confidence", 0)
    missing = result.get("missing_fields", [])
    cust_name = (result.get("customer") or {}).get("name")
    emp_first = (result.get("employee") or {}).get("first_name")

    has_parse_error = "parse_error" in (missing or [])
    task_ok = task == expected_task
    cust_ok = (expected_customer is None) or (cust_name == expected_customer)
    entity_ok = (expected_task == "create_employee" and emp_first) or cust_ok

    ok = task_ok and not has_parse_error and entity_ok
    status = "OK  " if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] {label}")
    print(f"       task={task!r} conf={conf:.2f} customer={cust_name!r} emp={emp_first!r} missing={missing}")

print(f"\n{passed}/{passed+failed} passed")
if failed:
    sys.exit(1)
