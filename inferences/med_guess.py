import csv
import os
from rapidfuzz import process, fuzz

# load medicine names (from CSV)
# structure → [["Paracetamol", "Fever"], ["Ibuprofen", "Pain relief"], ...]
medicines = []  

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "medicine_names.csv")

# read CSV once at startup 
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["Name"].strip()
        usage = row.get("Uses", "").strip()
        if name:  
            medicines.append([name, usage])  


 # lookup function
def find_med(query: str):
    """
    user will give med name and it will try to predict top 5 closest possibilities.
    """

    # pull out all names only
    names = [m[0] for m in medicines]

    # search top 5 matches 
    matches = process.extract(query, names, limit=5, scorer=fuzz.WRatio)

    # curating the results....
    results = [
        {
            "name": medicines[idx][0],   # original name
            "usage": medicines[idx][1],  # what it’s for (if available)
            "confi": round(score, 1)     # confidence score (0–100, roughly)
        }
        for _, score, idx in matches
    ]

    return results
