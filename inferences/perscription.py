import sqlite3
import random
from datetime import datetime, timedelta
import os


# using sqlite for database queries.
DB_NAME = os.path.join(os.path.dirname(__file__), "records.db")

def init_db():
    """Create table """
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id TEXT,  
        name TEXT NOT NULL,
        date TEXT NOT NULL,
        remarks TEXT,
        perscription TEXT
     )
     """)
    conn.commit()
    conn.close()


def add_record(record_id: str, name: str, date: str, remarks: str, perscription: str) -> int:
    """Adding a new record by id and returning true"""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO records (id, name, date, remarks, perscription) VALUES (?, ?, ?, ?,?)",
        (record_id, name, date, remarks, perscription)
    )
    conn.commit()
    conn.close()
    return 1


def fetch_by_id(record_id: str) -> dict[str, any]:
    """Fetch record by id"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  
    cur = conn.cursor()
    cur.execute("SELECT * FROM records WHERE id = ?", (record_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}


def delete_by_id(record_id: str) -> bool:
    """Delete by id"""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM records WHERE id = ? ORDER BY date DESC", (record_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return 1


def fetch_all_sorted_by_date(record_id: str) -> list[dict[str, any]]:
    """Fetch all records sorted by date"""
    # print("on my way to db")
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM records WHERE id = ? ORDER BY date ASC",(record_id,))
    rows = cur.fetchall()
    conn.close()
    records = [dict(row) for row in rows]
    return records


# -------- Adding random data to my db --------
if __name__ == "__main__":

 names = [
    "Ali Khan", "Umar Farooq", "Daud Ahmed", "Bashir Kasir",
    "Gul Shahnaz", "Ahmad Butt", "Sana Tariq", "Hina Malik",
    "Rashid Mehmood", "Ayesha Baloch"
   ]

 ids = [random.randint(1000, 9999) for _ in range(len(names))]

 remarks = [
    "Patient recovering well",
    "Needs more rest",
    "Prescribed antibiotics",
    "Blood pressure stable",
    "Diabetic checkup advised",
    "Follow up in 2 weeks",
    "Prescribed multivitamins",
    "Recommended physiotherapy",
    "Minor flu, prescribed syrup",
    "Headache complaints",
    "Routine checkup done",
    "No major issues found",
    "Patient has mild fever",
    "Advised blood test",
    "Prescribed painkillers",
    "Needs further diagnosis",
    "Patient reported dizziness",
    "Chest pain checkup",
    "Improvement noted",
    "Surgery follow-up"
 ]
 prescriptions = [
    "Amoxicillin 500mg - 3x/day",
    "Paracetamol 650mg - 2x/day",
    "Ibuprofen 400mg - 3x/day",
    "Metformin 500mg - 2x/day",
    "Amlodipine 5mg - 1x/day",
    "Cetirizine 10mg - 1x/day",
    "Omeprazole 20mg - 1x/day",
    "Azithromycin 250mg - 1x/day",
    "Lisinopril 10mg - 1x/day",
    "Vitamin D 1000 IU - 1x/day",
    "Insulin 10 units - before meals",
    "Prednisolone 5mg - 2x/day",
    "Ranitidine 150mg - 2x/day",
    "Hydrochlorothiazide 25mg - 1x/day",
    "Salbutamol inhaler - as needed",
    "Metoprolol 50mg - 2x/day",
    "Cefuroxime 500mg - 2x/day",
    "Levothyroxine 50mcg - 1x/day",
    "Furosemide 20mg - 1x/day",
    "Clarithromycin 500mg - 2x/day"
  ]


 
 #generate random date
 
 def random_date_2025():
    start = datetime(2025, 1, 1)
    end = datetime(2025, 9, 25)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")

 init_db()

 # the loop below is for random records is db

 for i, name in enumerate(names):
         for _ in range(10):  # 10 records per person
             date = random_date_2025()
             remark = random.choice(remarks)
             perscript = random.choice(prescriptions)
             add_record(ids[i], name, date, remark, perscript)


 # testing zone below

 conn = sqlite3.connect(DB_NAME)
 cur = conn.cursor()
 cur.execute("SELECT * FROM records ORDER BY date DESC LIMIT 10")
 rows = cur.fetchall()
 rows2 = fetch_all_sorted_by_date("6423")
 print(rows)
 print(rows2)



# print("Fetch by ID:", fetch_by_id(rid))
# print("All sorted by date:", fetch_all_sorted_by_date())

# print("Deleting record:", delete_by_id(rid))
# print("After delete:", fetch_all_sorted_by_date())