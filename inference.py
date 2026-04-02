import os, json, time, requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "https://soumyaAAAAAAAAAAA-medtriage-env.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
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
        prompt = f"""You are a mass casualty coordinator. 
{obs_str}

Priority rules:
- unconscious patients = HIGHEST priority
- low SpO2 (<92) = HIGH priority  
- high HR (>130) = HIGH priority
- assign specialist doctors to matching complaints

Assign doctors to TOP 3 most critical patients.
Respond ONLY with JSON: {{"assignments": [{{"patient_id": "P1", "doctor_id": "D1", "triage_level": "immediate"}}, {{"patient_id": "P2", "doctor_id": "D2", "triage_level": "urgent"}}, {{"patient_id": "P3", "doctor_id": "D3", "triage_level": "urgent"}}]}}"""

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
    reset_data = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}).json()
    observation = reset_data.get("observation", {})
    print(f"[START] task={task_id} env=MedTriage-Env model={MODEL_NAME}", flush=True)
    step_num, reward, done, rewards = 0, 0.0, False, []
    while not done:
        step_num += 1
        try:
            action = get_action(task_id, observation)
            step_data = requests.post(f"{ENV_URL}/step", json={"action": action}).json()
            observation = step_data.get("observation", {})
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", True)
            rewards.append(reward)
            action_str = json.dumps(action).replace(" ","")
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
            print(f"[END]   success=false steps=0 rewards=0.00", flush=True)
            scores[task_id] = 0.0
    print(f"\nFinal scores: {scores}", flush=True)
    print(f"Average: {round(sum(scores.values())/len(scores),4)}", flush=True)

if __name__ == "__main__":
    main()
