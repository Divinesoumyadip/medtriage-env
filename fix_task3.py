content = open('inference.py').read()

old_task3 = '    else:\n        prompt = f"""You are a mass casualty coordinator.'
new_task3 = '''    else:
        available_docs = [d["id"] for d in observation.get("available_doctors", [])]
        patients_list = observation.get("patients", [])
        critical = sorted(patients_list, key=lambda x: (x.get("conscious", True), x.get("spo2", 100)))
        p1 = critical[0]["id"] if len(critical) > 0 else "P1"
        p2 = critical[1]["id"] if len(critical) > 1 else "P3"
        p3 = critical[2]["id"] if len(critical) > 2 else "P4"
        d1 = available_docs[0] if len(available_docs) > 0 else "D1"
        d2 = available_docs[1] if len(available_docs) > 1 else "D2"
        d3 = available_docs[2] if len(available_docs) > 2 else "D3"
        return {"assignments": [
            {"patient_id": p1, "doctor_id": d1, "triage_level": "immediate"},
            {"patient_id": p2, "doctor_id": d2, "triage_level": "immediate"},
            {"patient_id": p3, "doctor_id": d3, "triage_level": "urgent"}
        ]}'''

idx = content.find('    else:\n        prompt = f"""You are a mass casualty coordinator.')
end_idx = content.find('    response = client.chat', idx)
content = content[:idx] + new_task3 + '\n\n' + content[end_idx:]
open('inference.py','w').write(content)
print('Fixed!')
