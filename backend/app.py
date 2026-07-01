from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from google import genai
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import os
import json
import uuid

print("STARTING FLASK FILE")

load_dotenv()

app = Flask(__name__)
CORS(app)


# =========================
# GEMINI SETUP
# =========================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Guard: if no API key, set client to None so we return clean fallback responses
# instead of crashing on every triage request
if not os.getenv("GEMINI_API_KEY", "").strip():
    client = None
    print("WARNING: GEMINI_API_KEY is not set. AI triage will use fallback responses.")

# =========================
# IN-MEMORY STORAGE
# =========================

patient_queue = []
active_beds = []
audit_log = []
chat_history = []

# =========================
# DATABASE
# =========================

import sqlite3

DB_NAME = "triage.db"


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS patients(
        id TEXT PRIMARY KEY,
        name TEXT,
        language TEXT,
        symptoms TEXT,
        created_at TEXT,
        updated_at TEXT,
        status TEXT,
        result_json TEXT,
        treatment_notes TEXT,
        doctor_override TEXT,
        history_json TEXT,
        arrival_time TEXT,
        treatment_started TEXT,
        discharged INTEGER,
        age INTEGER,
        patient_flag TEXT
    )
    """)
    conn.commit()

    # Self-migrate: if this is an older triage.db missing newer columns
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(patients)")}
    required_columns = {
        "age": "INTEGER",
        "patient_flag": "TEXT",
    }
    for column, col_type in required_columns.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE patients ADD COLUMN {column} {col_type}")
            print(f"Migrated triage.db: added missing column '{column}'")
    conn.commit()
    conn.close()


initialize_database()

# =========================
# TRIAGE PROMPT
# =========================

SYSTEM_PROMPT = """
You are an emergency-room triage assistant.

Analyze the patient's symptoms and return ONLY valid JSON.

Use ESI levels:

ESI-1 = Immediate life threat
ESI-2 = Emergent
ESI-3 = Urgent
ESI-4 = Less Urgent
ESI-5 = Non-Urgent

Return exactly:

{
  "translation":"",
  "anatomy":[],
  "vitals":[],
  "allergies":[],
  "medications":[],
  "alerts":[],
  "esi":"",
  "esi_label":"",
  "confidence":0.0,
  "rationale":"",
  "patient_instructions":"",
  "wait_time_estimate":"",
  "requires_human_review":false
}

Rules:
- Detect the patient's language reliably.
- Translate symptoms into clear medical English.
- Preserve medical meaning (do NOT literal-translate slang if it changes meaning).
- If language is unclear, infer from context.
- Extract anatomy/body parts.
- Extract medications.
- Extract allergies.
- Extract severity clues.
- Generate a rationale.
- Generate patient instructions.
- Confidence must be 0.0 to 1.0.
- Return JSON only.
"""

DOCTOR_ASSISTANT_PROMPT = """
You are a medical chatbot for an emergency department.

You have access to the live patient queue, active beds, and each patient's full condition timeline.

Use conversation history to maintain context across questions.

Prioritize by ESI level, flags (priority/extended/deceased), and waiting time.

Be concise, clinically useful, and reference specific patients by name when relevant.

If asked about trends, compare timeline entries. If asked for a summary, cover all flagged and critical patients first.
"""

# =========================
# GEMINI ANALYSIS
# =========================

def analyze_with_gemini(language, symptoms):
    prompt = f"""
Patient Language: {language}

Patient Statement:
{symptoms}
"""
    if client is None:
        return {
            "translation": symptoms,
            "anatomy": [], "vitals": [], "allergies": [], "medications": [],
            "alerts": ["Gemini API key not configured"],
            "esi": "ESI-3", "esi_label": "Level 3: Urgent",
            "confidence": 0.5,
            "rationale": "Set GEMINI_API_KEY in your .env to enable AI triage.",
            "patient_instructions": "Please wait for medical staff.",
            "wait_time_estimate": "30-60 minutes",
            "requires_human_review": True
        }

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}"
        )
    except Exception as e:
        return {
            "translation": symptoms,
            "anatomy": [],
            "vitals": [],
            "allergies": [],
            "medications": [],
            "alerts": ["Gemini API error"],
            "esi": "ESI-3",
            "esi_label": "Level 3: Urgent",
            "confidence": 0.5,
            "rationale": str(e),
            "patient_instructions": "Please wait for medical staff.",
            "wait_time_estimate": "30-60 minutes",
            "requires_human_review": True
        }

    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(text)
    except Exception:
        return {
            "translation": symptoms,
            "anatomy": [],
            "vitals": [],
            "allergies": [],
            "medications": [],
            "alerts": ["Invalid AI response"],
            "esi": "ESI-3",
            "esi_label": "Level 3: Urgent",
            "confidence": 0.5,
            "rationale": text,
            "patient_instructions": "Please wait for medical staff.",
            "wait_time_estimate": "30-60 minutes",
            "requires_human_review": True
        }

    if result.get("confidence", 1.0) < 0.75:
        result["requires_human_review"] = True

    return result

# =========================
# HELPERS
# =========================

def esi_priority(esi):
    mapping = {
        "ESI-1": 1,
        "ESI-2": 2,
        "ESI-3": 3,
        "ESI-4": 4,
        "ESI-5": 5
    }
    return mapping.get(esi, 5)


def sort_queue():
    patient_queue.sort(
        key=lambda p: (
            int(p["result"]["esi"].split("-")[1]) if "-" in p["result"].get("esi", "") else 5,
            -p.get("waiting_minutes", 0)
        )
    )


def log_event(action, patient):
    audit_log.append({
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "patient": patient["name"],
        "status": patient["status"],
        "esi": patient["result"]["esi"]
    })

# =========================
# WAIT TIME
# =========================

def waiting_minutes(patient):
    arrival_str = patient.get("arrival_time")
    if not arrival_str:
        return 0
    try:
        arrival = datetime.fromisoformat(arrival_str)
        return int((datetime.now() - arrival).total_seconds() / 60)
    except Exception:
        return 0


def update_wait_times():
    for patient in patient_queue:
        patient["waiting_minutes"] = waiting_minutes(patient)
        save_patient(patient)

# =========================
# SAVE PATIENT
# =========================

def save_patient(patient):
    conn = get_db()
    conn.execute(
        """
        INSERT OR REPLACE INTO patients
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            patient["id"],
            patient["name"],
            patient["language"],
            patient["symptoms"],
            patient["created_at"],
            patient["updated_at"],
            patient["status"],
            json.dumps(patient["result"]),
            patient.get("treatment_notes", ""),
            json.dumps(patient.get("doctor_override")),
            json.dumps(patient.get("history", [])),
            patient.get("arrival_time", ""),
            patient.get("treatment_started"),
            1 if patient.get("discharged") else 0,
            patient.get("age"),
            patient.get("patient_flag", "")
        )
    )
    conn.commit()
    conn.close()

# =========================
# LOAD PATIENTS
# =========================

def load_patients():
    global patient_queue, active_beds, audit_log

    conn = get_db()
    rows = conn.execute("SELECT * FROM patients").fetchall()
    conn.close()

    patient_queue.clear()
    active_beds.clear()
    audit_log.clear()

    for row in rows:
        patient = {
            "id": row["id"],
            "name": row["name"],
            "language": row["language"],
            "symptoms": row["symptoms"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "result": json.loads(row["result_json"]),
            "treatment_notes": row["treatment_notes"] or "",
            "doctor_override": json.loads(row["doctor_override"]) if row["doctor_override"] else None,
            "history": json.loads(row["history_json"]) if row["history_json"] else [],
            "arrival_time": row["arrival_time"] or "",
            "treatment_started": row["treatment_started"],
            "discharged": bool(row["discharged"]),
            "age": row["age"],
            "patient_flag": row["patient_flag"] or ""
        }
        patient["waiting_minutes"] = waiting_minutes(patient)

        if patient["status"] == "In Treatment":
            active_beds.append(patient)
        elif patient["status"] != "Discharged":
            patient_queue.append(patient)

    sort_queue()

# =========================
# HOME
# =========================

@app.route("/")
def home():
    return render_template("index.html")

# =========================
# HEALTH CHECK
# =========================

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model": "gemini-2.5-flash",
        "queue": len(patient_queue),
        "treatment": len(active_beds),
        "database": "SQLite",
        "chat_memory": len(chat_history)
    })

# =========================
# TRIAGE PATIENT
# =========================

@app.route("/analyze", methods=["POST"])
def analyze():
    data = None
    try:
        data = request.get_json()
        name = data.get("name", "")
        language = data.get("language", "")
        symptoms = data.get("symptoms", "")
        age = data.get("age", None)
        patient_flag = data.get("patient_flag", "")

        result = analyze_with_gemini(language, symptoms)

        now = datetime.now().isoformat()
        patient = {
            "id": str(uuid.uuid4()),
            "name": name,
            "language": language,
            "symptoms": symptoms,
            "created_at": now,
            "updated_at": now,
            "status": "Waiting",
            "result": result,
            "history": [result],
            "treatment_notes": "",
            "doctor_override": None,
            "arrival_time": now,
            "treatment_started": None,
            "discharged": False,
            "waiting_minutes": 0,
            "age": age,
            "patient_flag": patient_flag
        }

        patient_queue.append(patient)
        sort_queue()
        save_patient(patient)

        audit_log.append({
            "timestamp": now,
            "action": "Triage",
            "patient": name,
            "status": "Waiting",
            "esi": result.get("esi", "ESI-3")
        })

        return jsonify(result)

    except Exception as e:
        symptoms_fallback = data.get("symptoms", "") if data else ""
        return jsonify({
            "translation": symptoms_fallback,
            "anatomy": [],
            "vitals": [],
            "allergies": [],
            "medications": [],
            "alerts": ["Gemini unavailable"],
            "esi": "ESI-3",
            "esi_label": "Level 3: Urgent",
            "confidence": 0.50,
            "rationale": str(e),
            "patient_instructions": "Please wait for medical staff.",
            "wait_time_estimate": "30-60 minutes",
            "requires_human_review": True
        })

# =========================
# LIVE QUEUE
# =========================

@app.route("/queue")
def queue():
    update_wait_times()
    sort_queue()
    return jsonify(patient_queue)

# =========================
# ACTIVE BEDS
# =========================

@app.route("/beds")
def beds():
    return jsonify(active_beds)

# =========================
# MOVE TO BED
# =========================

@app.route("/move_to_bed", methods=["POST"])
def move_to_bed():
    data = request.get_json()
    patient_id = data["id"]

    for patient in patient_queue:
        if patient["id"] == patient_id:
            patient_queue.remove(patient)
            patient["status"] = "In Treatment"
            patient["treatment_started"] = datetime.now().isoformat()
            active_beds.append(patient)
            save_patient(patient)
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "Moved To Treatment",
                "patient": patient["name"],
                "status": "In Treatment",
                "esi": patient["result"]["esi"]
            })
            return jsonify({"message": "Patient moved to treatment."})

    return jsonify({"error": "Patient not found."}), 404

# =========================
# RETRIAGE
# =========================

@app.route("/recheck", methods=["POST"])
def recheck():
    data = request.get_json()
    patient_id = data["id"]
    updated = data["symptoms"]

    for patient in patient_queue:
        if patient["id"] == patient_id:
            patient["history"].append(patient["result"])
            patient["symptoms"] = updated
            patient["updated_at"] = datetime.now().isoformat()
            patient["result"] = analyze_with_gemini(patient["language"], updated)
            sort_queue()
            save_patient(patient)
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "Re-Triage",
                "patient": patient["name"],
                "status": patient["status"],
                "esi": patient["result"]["esi"]
            })
            return jsonify(patient["result"])

    return jsonify({"error": "Patient not found"}), 404

# =========================
# CRITICAL CASES
# =========================

@app.route("/critical")
def critical():
    critical_cases = [
        p for p in patient_queue
        if p["result"]["esi"] in ["ESI-1", "ESI-2"]
    ]
    return jsonify(critical_cases)

# =========================
# DOCTOR OVERRIDE
# =========================

@app.route("/override", methods=["POST"])
def override():
    data = request.get_json()
    patient_id = data["id"]
    new_esi = data["esi"]
    reason = data["reason"]
    doctor = data.get("doctor", "Unknown")

    for patient in patient_queue:
        if patient["id"] == patient_id:
            old = patient["result"]["esi"]
            patient["doctor_override"] = {
                "old": old,
                "new": new_esi,
                "reason": reason,
                "doctor": doctor,
                "timestamp": datetime.now().isoformat()
            }
            patient["result"]["esi"] = new_esi
            sort_queue()
            save_patient(patient)
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "Doctor Override",
                "patient": patient["name"],
                "status": patient["status"],
                "esi": new_esi
            })
            return jsonify({"message": "Override Saved"})

    return jsonify({"error": "Patient not found"}), 404

# =========================
# AUDIT LOG
# =========================

@app.route("/audit")
def audit():
    return jsonify({"events": audit_log[-100:]})

# =========================
# ANALYTICS
# =========================

@app.route("/analytics")
def analytics():
    update_wait_times()

    language_counts = {}
    esi_counts = {
        "ESI-1": 0, "ESI-2": 0, "ESI-3": 0,
        "ESI-4": 0, "ESI-5": 0
    }

    total_wait = 0
    total_confidence = 0
    critical_cases = 0
    oldest_wait = 0
    highest_priority = None

    all_patients = patient_queue + active_beds

    for patient in all_patients:
        language = patient["language"] or "Unknown"
        language_counts[language] = language_counts.get(language, 0) + 1

        esi = patient["result"].get("esi", "ESI-5")
        if esi in esi_counts:
            esi_counts[esi] += 1

        if esi in ["ESI-1", "ESI-2"]:
            critical_cases += 1

        total_confidence += patient["result"].get("confidence", 0)

    for patient in patient_queue:
        wait = patient.get("waiting_minutes", 0)
        total_wait += wait
        if wait > oldest_wait:
            oldest_wait = wait

    if patient_queue:
        sort_queue()
        highest_priority = {
            "name": patient_queue[0]["name"],
            "esi": patient_queue[0]["result"].get("esi", "ESI-5"),
            "waiting_minutes": patient_queue[0].get("waiting_minutes", 0)
        }

    treatment_minutes = 0
    for patient in active_beds:
        if patient.get("treatment_started"):
            try:
                start = datetime.fromisoformat(patient["treatment_started"])
                treatment_minutes += int(
                    (datetime.now() - start).total_seconds() / 60
                )
            except Exception:
                pass

    average_wait = total_wait / len(patient_queue) if patient_queue else 0
    average_confidence = total_confidence / len(all_patients) if all_patients else 0
    average_treatment = treatment_minutes / len(active_beds) if active_beds else 0

    conn = get_db()
    discharged = conn.execute(
        "SELECT COUNT(*) FROM patients WHERE status='Discharged'"
    ).fetchone()[0]
    conn.close()

    return jsonify({
        "patients_waiting": len(patient_queue),
        "patients_in_treatment": len(active_beds),
        "critical_cases": critical_cases,
        "average_wait": round(average_wait, 1),
        "oldest_wait": oldest_wait,
        "average_confidence": round(average_confidence, 2),
        "average_treatment_minutes": round(average_treatment, 1),
        "patients_discharged": discharged,
        "languages": language_counts,
        "esi_distribution": esi_counts
    })

# =========================
# TEXT TO SPEECH
# =========================

@app.route("/speak")
def speak():
    text = request.args.get("text", "Please wait for medical staff.")
    return jsonify({"text": text})

# =========================
# PICTURE SYMPTOMS
# =========================

@app.route("/symptom_icons")
def symptom_icons():
    return jsonify([
        {"name": "Chest Pain",         "emoji": "❤️"},
        {"name": "Headache",           "emoji": "🧠"},
        {"name": "Breathing Problem",  "emoji": "🫁"},
        {"name": "Stomach Pain",       "emoji": "🤢"},
        {"name": "Broken Bone",        "emoji": "🦴"},
        {"name": "Fever",              "emoji": "🤒"}
    ])

# =========================
# AI ASSISTANT
# =========================

@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json()
    question = data.get("question", "")

    def fmt_patient(p):
        flag = p.get("patient_flag", "")
        age = p.get("age")
        tl = [e.get("note","") or e.get("esi","") for e in p.get("history", [])]
        return f"""
Name: {p['name']}{f' (Age: {age})' if age else ''}
Status: {p['status']}{f' | FLAG: {flag.upper()}' if flag else ''}
Language: {p['language']}
Waiting: {p.get('waiting_minutes', 0)} minutes
Symptoms: {p['symptoms']}
ESI: {p['result'].get('esi','')} | Confidence: {p['result'].get('confidence','')}
Rationale: {p['result'].get('rationale','')}
Timeline entries: {'; '.join(tl[-5:]) if tl else 'none'}
Treatment Notes: {p.get('treatment_notes','none')}
"""

    queue_text = "=== WAITING ===\n"
    for patient in patient_queue:
        queue_text += fmt_patient(patient)

    queue_text += "\n=== IN TREATMENT ===\n"
    for patient in active_beds:
        queue_text += fmt_patient(patient)

    conversation = ""
    for msg in chat_history[-20:]:
        conversation += f"\n{msg['role']}: {msg['text']}\n"

    prompt = f"""
Hospital State:
{queue_text}

Conversation:
{conversation}

Doctor: {question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{DOCTOR_ASSISTANT_PROMPT}\n\n{prompt}"
    )

    answer = response.text

    chat_history.append({"role": "Doctor", "text": question})
    chat_history.append({"role": "Assistant", "text": answer})

    return jsonify({"answer": answer})

# =========================
# TREATMENT NOTES
# =========================

@app.route("/treatment_notes", methods=["POST"])
def treatment_notes():
    data = request.get_json()
    patient_id = data["id"]
    notes = data["notes"]

    for patient in active_beds:
        if patient["id"] == patient_id:
            patient["treatment_notes"] = notes
            patient["updated_at"] = datetime.now().isoformat()
            save_patient(patient)
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "Treatment Notes Updated",
                "patient": patient["name"],
                "status": patient["status"],
                "esi": patient["result"]["esi"]
            })
            return jsonify({"message": "Saved"})

    return jsonify({"error": "Patient not found"}), 404

# =========================
# DISCHARGE
# =========================

@app.route("/discharge", methods=["POST"])
def discharge():
    data = request.get_json()
    patient_id = data["id"]

    for patient in active_beds:
        if patient["id"] == patient_id:
            active_beds.remove(patient)
            patient["status"] = "Discharged"
            patient["discharged"] = True
            patient["updated_at"] = datetime.now().isoformat()
            save_patient(patient)
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "Discharged",
                "patient": patient["name"],
                "status": "Discharged",
                "esi": patient["result"]["esi"]
            })
            return jsonify({"message": "Patient discharged."})

    return jsonify({"error": "Patient not found."}), 404

# =========================
# CRITICAL BANNER
# =========================

@app.route("/critical_banner")
def critical_banner():
    update_wait_times()
    sort_queue()

    for patient in patient_queue:
        if patient["result"].get("esi") in ["ESI-1", "ESI-2"]:
            return jsonify({
                "critical": True,
                "patient": patient,
                "message": f"{patient['name']} requires immediate attention."
            })

    return jsonify({"critical": False, "patient": None, "message": ""})

# =========================
# TIMELINE GET
# =========================

@app.route("/timeline/<patient_id>")
def timeline(patient_id):
    for patient in patient_queue + active_beds:
        if patient["id"] == patient_id:
            return jsonify({
                "id": patient["id"],
                "name": patient["name"],
                "history": patient["history"],
                "override": patient["doctor_override"],
                "treatment_notes": patient["treatment_notes"],
                "status": patient["status"]
            })
    return jsonify({"error": "Patient not found"}), 404

# =========================
# TIMELINE POST (log condition)
# =========================

@app.route("/timeline/<patient_id>", methods=["POST"])
def timeline_log(patient_id):
    data = request.get_json()
    note = data.get("note", "").strip()
    if not note:
        return jsonify({"error": "Note is required"}), 400

    entry = {
        "type": "condition_log",
        "timestamp": datetime.now().isoformat(),
        "note": note
    }

    for patient in patient_queue + active_beds:
        if patient["id"] == patient_id:
            patient["history"].append(entry)
            patient["updated_at"] = datetime.now().isoformat()
            save_patient(patient)
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "Condition Logged",
                "patient": patient["name"],
                "status": patient["status"],
                "esi": patient["result"]["esi"]
            })
            return jsonify({"message": "Logged", "entry": entry})

    return jsonify({"error": "Patient not found"}), 404

# =========================
# SEARCH
# =========================

@app.route("/search")
def search():
    query = request.args.get("q", "").lower()
    matches = [
        patient for patient in patient_queue + active_beds
        if (
            query in patient["name"].lower()
            or query in patient["symptoms"].lower()
            or query in (patient["language"] or "").lower()
        )
    ]
    return jsonify(matches)

# =========================
# CLEAR CHAT
# =========================

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    chat_history.clear()
    return jsonify({"message": "Chat cleared."})

# =========================
# NOTIFICATIONS
# =========================

@app.route("/notifications")
def notifications():
    alerts = []
    for patient in patient_queue:
        if patient["result"].get("esi") in ["ESI-1", "ESI-2"]:
            alerts.append({
                "type": "critical",
                "patient": patient["name"],
                "message": "Critical patient waiting."
            })
        if patient["result"].get("confidence", 1.0) < 0.60:
            alerts.append({
                "type": "low_confidence",
                "patient": patient["name"],
                "message": "AI confidence is low."
            })
    return jsonify(alerts)

# =========================
# DASHBOARD
# =========================

@app.route("/dashboard")
def dashboard():
    update_wait_times()
    return jsonify({
        "waiting": len(patient_queue),
        "treatment": len(active_beds),
        "alerts": len([
            p for p in patient_queue
            if p["result"].get("esi") in ["ESI-1", "ESI-2"]
        ]),
        "audit_events": len(audit_log)
    })

# =========================
# ARCHIVE
# =========================

@app.route("/archive")
def archive():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM patients WHERE status='Discharged'"
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

# =========================
# WIPE ALL DATA
# =========================

@app.route("/wipe_all_data", methods=["POST"])
def wipe_all_data():
    conn = get_db()
    conn.execute("DELETE FROM patients")
    conn.commit()
    conn.close()

    patient_queue.clear()
    active_beds.clear()
    audit_log.clear()
    chat_history.clear()

    return jsonify({"message": "All patient data wiped."})

# =========================
# PATIENT FLAG
# =========================

@app.route("/flag", methods=["POST"])
def flag_patient():
    data = request.get_json()
    patient_id = data["id"]
    flag = data.get("flag", "")  # "extended", "priority", "deceased", ""

    for patient in patient_queue + active_beds:
        if patient["id"] == patient_id:
            patient["patient_flag"] = flag
            patient["updated_at"] = datetime.now().isoformat()
            save_patient(patient)
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": f"Flag: {flag or 'cleared'}",
                "patient": patient["name"],
                "status": patient["status"],
                "esi": patient["result"]["esi"]
            })
            return jsonify({"message": "Flag updated", "flag": flag})

    return jsonify({"error": "Patient not found"}), 404

@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({"error": str(error)}), 500

if __name__ == "__main__":
    load_patients()
    print("ABOUT TO START SERVER")
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )
