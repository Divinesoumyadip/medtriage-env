import random
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
import uvicorn

app = FastAPI(title="MedTriage-Env")

TRIAGE_LEVELS = ["minor", "delayed", "urgent", "immediate"]

PATIENTS = [
    {"id":"P1","name":"Male 45","bp":"80/40","hr":145,"spo2":88,"conscious":False,"complaint":"chest pain","true_level":"immediate","specialty":"cardiology"},
    {"id":"P2","name":"Female 32","bp":"120/80","hr":95,"spo2":97,"conscious":True,"complaint":"mild fever","true_level":"minor","specialty":"general"},
    {"id":"P3","name":"Child 8","bp":"90/60","hr":130,"spo2":91,"conscious":True,"complaint":"difficulty breathing","true_level":"urgent","specialty":"pediatrics"},
    {"id":"P4","name":"Male 67","bp":"100/70","hr":110,"spo2":93,"conscious":True,"complaint":"stroke symptoms","true_level":"immediate","specialty":"neurology"},
    {"id":"P5","name":"Female 25","bp":"110/75","hr":88,"spo2":96,"conscious":True,"complaint":"broken arm","true_level":"delayed","specialty":"orthopedics"},
    {"id":"P6","name":"Male 55","bp":"85/50","hr":150,"spo2":85,"conscious":False,"complaint":"internal bleeding","true_level":"immediate","specialty":"surgery"},
    {"id":"P7","name":"Female 40","bp":"130/85","hr":78,"spo2":98,"conscious":True,"complaint":"headache","true_level":"minor","specialty":"general"},
    {"id":"P8","name":"Male 70","bp":"95/65","hr":120,"spo2":90,"conscious":True,"complaint":"severe abdominal pain","true_level":"urgent","specialty":"surgery"},
]

DOCTORS = [
    {"id":"D1","name":"Dr. Sharma","specialty":"cardiology","available":True},
    {"id":"D2","name":"Dr. Patel","specialty":"general","available":True},
    {"id":"D3","name":"Dr. Khan","specialty":"surgery","available":True},
    {"id":"D4","name":"Dr. Roy","specialty":"pediatrics","available":True},
    {"id":"D5","name":"Dr. Mehta","specialty":"neurology","available":True},
]

env_state = {"task_id":None,"current_patient":None,"multi_patients":[],"doctors":[],"step_count":0,"done":False,"score":0.0,"deterioration_timer":0}

class ResetRequest(BaseModel):
    task_id: str = "task1"

    @classmethod
    def model_validate(cls, obj):
        if obj is None: obj = {}
        return super().model_validate(obj)
    model_config = {"extra": "allow"}

class StepAction(BaseModel):
    action: dict

def triage_score(predicted, true_level, confidence=1.0):
    if predicted not in TRIAGE_LEVELS: return 0.0
    if predicted == true_level: base = 1.0
    else:
        diff = abs(TRIAGE_LEVELS.index(predicted) - TRIAGE_LEVELS.index(true_level))
        base = max(0.0, 1.0 - diff * 0.3)
    if predicted != true_level and confidence > 0.8: base *= 0.5
    return round(base * min(confidence + 0.2, 1.0), 2)

def assign_score(patient, doctor_id, doctors_list):
    doc = next((d for d in doctors_list if d["id"] == doctor_id), None)
    if not doc or not doc["available"]: return 0.0
    if doc["specialty"] == patient["specialty"]: return 1.0
    if doc["specialty"] == "general": return 0.6
    return 0.3

def deterioration_penalty(steps, true_level):
    if true_level == "immediate" and steps > 2: return max(0.0, 1.0 - (steps-2)*0.2)
    if true_level == "urgent" and steps > 3: return max(0.0, 1.0 - (steps-3)*0.15)
    return 1.0

@app.get("/")
def root(): return {"status":"ok","env":"MedTriage-Env","tasks":["task1","task2","task3"]}

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/tasks")
def tasks(): return {"tasks":[{"id":"task1","name":"Vital Signs Triage","difficulty":"easy"},{"id":"task2","name":"Doctor Assignment","difficulty":"medium"},{"id":"task3","name":"Mass Casualty Coordination","difficulty":"hard"}]}

@app.get("/state")
def state(): return {"task_id":env_state["task_id"],"step_count":env_state["step_count"],"done":env_state["done"],"score":env_state["score"]}

@app.post("/reset")
def reset(req: ResetRequest = None):
    if req is None: req = ResetRequest()
    global env_state
    env_state.update({"task_id":req.task_id,"step_count":0,"done":False,"score":0.0,"deterioration_timer":0})
    doctors = [d.copy() for d in DOCTORS]
    env_state["doctors"] = doctors
    if req.task_id == "task1":
        p = random.choice(PATIENTS)
        env_state["current_patient"] = p
        return {"observation":{"patient_id":p["id"],"vitals":{"bp":p["bp"],"hr":p["hr"],"spo2":p["spo2"]},"conscious":p["conscious"],"complaint":p["complaint"],"instructions":"Action: {triage_level: 'immediate|urgent|delayed|minor', confidence: 0.0-1.0}"},"reward":0.0,"done":False,"info":{"task":"task1"}}
    elif req.task_id == "task2":
        p = random.choice(PATIENTS)
        env_state["current_patient"] = p
        return {"observation":{"patient_id":p["id"],"complaint":p["complaint"],"vitals":{"bp":p["bp"],"hr":p["hr"],"spo2":p["spo2"]},"available_doctors":[d for d in doctors if d["available"]],"instructions":"Action: {assign_doctor_id: 'D1|D2|D3|D4|D5'}"},"reward":0.0,"done":False,"info":{"task":"task2"}}
    else:
        patients = random.sample(PATIENTS, 5)
        env_state["multi_patients"] = patients
        return {"observation":{"patients":[{"id":p["id"],"complaint":p["complaint"],"hr":p["hr"],"spo2":p["spo2"],"conscious":p["conscious"]} for p in patients],"available_doctors":[d for d in doctors if d["available"]][:3],"icu_beds":2,"instructions":"Action: {assignments:[{patient_id,doctor_id,triage_level}]} max 3"},"reward":0.0,"done":False,"info":{"task":"task3"}}

@app.post("/step")
def step(action: StepAction):
    global env_state
    if env_state["done"]: return {"observation":{},"reward":0.0,"done":True,"info":{}}
    env_state["step_count"] += 1
    act = action.action
    task = env_state["task_id"]
    if task == "task1":
        predicted = act.get("triage_level","")
        confidence = float(act.get("confidence", 0.8))
        true_level = env_state["current_patient"]["true_level"]
        det = deterioration_penalty(env_state["step_count"], true_level)
        reward = triage_score(predicted, true_level, confidence) * det
        env_state.update({"score":reward,"done":True})
        return {"observation":{"predicted":predicted,"true_level":true_level,"confidence":confidence},"reward":round(reward,2),"done":True,"info":{"score":round(reward,2)}}
    elif task == "task2":
        doc_id = act.get("assign_doctor_id","")
        patient = env_state["current_patient"]
        reward = assign_score(patient, doc_id, env_state["doctors"])
        for d in env_state["doctors"]:
            if d["id"] == doc_id: d["available"] = False
        env_state.update({"score":reward,"done":True})
        return {"observation":{"assigned_doctor":doc_id,"patient_specialty":patient["specialty"]},"reward":round(reward,2),"done":True,"info":{"score":round(reward,2)}}
    else:
        assignments = act.get("assignments",[])[:3]
        pm = {p["id"]:p for p in env_state["multi_patients"]}
        dm = {d["id"]:d for d in env_state["doctors"]}
        used = set()
        total = 0.0
        for a in assignments:
            pid,did,tlevel = a.get("patient_id",""),a.get("doctor_id",""),a.get("triage_level","")
            if pid not in pm or did not in dm or did in used: continue
            used.add(did)
            p = pm[pid]
            ts = triage_score(tlevel, p["true_level"])
            as_ = assign_score(p, did, env_state["doctors"])
            sw = (TRIAGE_LEVELS.index(p["true_level"])+1)/len(TRIAGE_LEVELS)
            det = deterioration_penalty(env_state["step_count"], p["true_level"])
            total += sw * ts * as_ * det
        reward = round(min(total/3.0,1.0),2)
        env_state.update({"score":reward,"done":True})
        return {"observation":{"processed":len(assignments)},"reward":reward,"done":True,"info":{"score":reward}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
