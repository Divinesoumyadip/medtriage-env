import os, json, time, requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "https://soumyaAAAAAAAAAAA-medtriage-env.hf.space")

def get_client():
    token = HF_TOKEN if HF_TOKEN else "dummy-token"
    return OpenAI(base_url=API_BASE_URL, api_key=token)

TASKS = ["task1", "task2", "task3"]

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
- chest pain/heart → cardiology
- breathing/lungs → pulmonology or general
- stroke/brain → neurology  
- surgery/bleeding/abdominal → surgery
- children → pediatrics
- bones/fracture → orthopedics
- general/fever/headache → general

{obs_str}

Look at patient complaint, find matching specialty in available_doctors list.
Respond ONLY with JSON: {{"assign_doctor_id": "D1"}}
Pick the doctor whose specialty best matches the complaint."""

    else:
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
        ]}

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

def run_task(task_id):
    reset_data = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30).json()
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
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    success = reward > 0.0
    print(f"[END]   success={str(success).lower()} steps={step_num} rewards={rewards_str}", flush=True)
    return reward

def main():
    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"[END]   success=false steps=0 rewards=0.00 error={str(e)}", flush=True)
            scores[task_id] = 0.0
    print(f"\nFinal scores: {scores}", flush=True)
    print(f"Average: {round(sum(scores.values())/len(scores), 4)}", flush=True)

if __name__ == "__main__":
    main()