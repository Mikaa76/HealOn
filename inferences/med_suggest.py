import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# load pre-trained model
base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "models", "med_suggest_model.pkl")
with open(model_path, "rb") as f:
    model = joblib.load(f)  


def recommend_by_symp(query: str, top_k: int = 5):
    """Suggest medicines based on a symptom/description query."""
    data, vectorizer, vectors = model["data"], model["vectorizer"], model["vectors"]

    # compare against all medicine vectors
    query_vec = vectorizer.transform([query.lower()])
    similarity = cosine_similarity(query_vec, vectors).flatten()

    query_tokens = set(query.split())
    results = []
    for idx, score in enumerate(similarity):
        row = data.iloc[idx]

        # extra weight if symptom words overlap with medicine uses
        overlap = len(query_tokens & set(row['Uses'].split()))
        final_score = score + 0.1 * overlap

        if final_score > 0.3:  # ignore weak matches
            results.append({
                "name": row['Name'],
                "url": row['Url'],
                "usage": row['Uses'],
                "confi": round(float(final_score * 100), 1)
            })

    return sorted(results, key=lambda x: x['confi'], reverse=True)[:top_k]


def recommend_by_medicine(med_name: str, top_k: int = 5):
    """Suggest alternatives for a known medicine name."""
    data = model["data"]
    match = data[data['Name'] == med_name.lower()]

    if match.empty:
        return [{"name": "404", "usage": "No such medicine found in database"}]

    # if found medicine in database, we will get its uses and look for similar medicines having similar uses.
    return recommend_by_symp(match.iloc[0]['Uses'], top_k=top_k)


# for testing.. when we run file directly
if __name__ == "__main__":
    print(">>> Symptom-based recommendation:")
    for r in recommend_by_symp("nose flu"):
        print(f"- {r['name']} | {r['usage']} : {r['confi']}")

    print("\n>>> Medicine-based recommendation:")
    for r in recommend_by_medicine("Panadol"):
        print(f"- {r['name']} | {r['usage']} : {r.get('confi')}")
