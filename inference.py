import os, json, time, requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.environ.get("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or "dummy-key"
ENV_URL      = os.environ.get("ENV_URL") or "https://soumyaAAAAAAAAAAA-medtriage-env.hf.space"

TASKS = ["task1", "task2", "task3"]

def get_client():
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def get_action(task_id, observation):
    obs_str = json.dumps(observation, indent=2)
    if task_id == "task1":
        prompt = f"""You are an expert ER triage nurse with 20 years experience.
Analyze these patient vitals carefully:
{obs_str}

Rules:
- immediate: unconscious, BP<90/60, HR>140, SpO2<90, chest pain, stroke, bleeding
- urgent: HR>120, SpO2<94, serious pain
- delayed: stable vitals, non-life-threatening
- minor: normal vitals, minor complaints

Respond ONLY with JSON: {{"triage_level": "immediate", "confidence": 0.95}}"""

    elif task_id == "task2":
        prompt = f"""You are an ER coordinator. Match patient to best doctor by specialty.

Specialty matching rules:
- chest pain/heart -> cardiology
- breathing/lungs -> pulmonology or general
- stroke/brain -> neurology
- surgery/bleeding/abdominal -> surgery
- children -> pediatrics
- bones/fracture -> orthopedics
- general/fever/headache -> general

{obs_str}

Look at patient complaint, find matching specialty in available_doctors list.
Respond ONLY with JSON: {{"assign_doctor_id": "D1"}}
Pick the doctor whose specialty best matches the complaint."""

    else:
        available_docs = [d["id"] for d in observation.get("available_doctors", [])]
        patients_list = observation.get("patients", [])
        critical = sorted(patients_list, key=lambda x: (x.get("conscious", True), x.get("spo2", 100)))
        assignments = []
        for i in range(min(3, len(critical), len(available_docs))):
            triage = "immediate" if i < 2 else "urgent"
            assignments.append({"patient_id": critical[i]["id"], "doctor_id": available_docs[i], "triage_level": triage})
        if not assignments:
            assignments = [{"patient_id": "P1", "doctor_id": "D1", "triage_level": "immediate"}]
        return {"assignments": assignments}

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME, max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"): text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"LLM Error: {e}", flush=True)
        if task_id == "task1": return {"triage_level": "immediate", "confidence": 0.90}
        if task_id == "task2": return {"assign_doctor_id": "D1"}
        return {"assignments": []}

def run_task(task_id):
    try:
        reset_data = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30).json()
    except Exception:
        return 0.0
    observation = reset_data.get("observation", {})
    print(f"[START] task={task_id} env=MedTriage-Env model={MODEL_NAME}", flush=True)
    step_num, reward, done, rewards = 0, 0.0, False, []
    while not done:
        step_num += 1
        try:
            action = get_action(task_id, observation)
            step_data = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30).json()
            observation = step_data.get("observation", {})
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", True)
            rewards.append(reward)
            action_str = json.dumps(action).replace(" ", "")
            print(f"[STEP]  step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        except Exception as e:
            rewards.append(0.0)
            print(f"[STEP]  step={step_num} action=null reward=0.00 done=true error={str(e)}", flush=True)
            done = True
        if done: break
    score = max(0.01, min(0.99, float(reward)))
    print(f"[END] task={task_id} score={score:.2f} steps={step_num}", flush=True)
    return reward

def main():
    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"[END] task={task_id} score=0.01 steps=0", flush=True)
            scores[task_id] = 0.0
    print(f"\nFinal scores: {scores}", flush=True)
    print(f"Average: {round(sum(scores.values())/len(scores), 4) if scores else 0}", flush=True)

if __name__ == "__main__":
    main()
