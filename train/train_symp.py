
# This file to create a model for medecine recommender based on symptoms etc


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class MedicineRecommender:
    def __init__(self, data):
        self.data = data[['Name', 'Uses', 'Url']].copy()
        self.data['Uses'] = self.data['Uses'].fillna("").astype(str).str.lower()
        self.data['Name'] = self.data['Name'].astype(str).str.lower()
        
        # Vectorizer + matrix
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.vectors = self.vectorizer.fit_transform(self.data['Uses'])
    
    def recommend(self, query, top_k=5):
     query = query.lower()
     query_vec = self.vectorizer.transform([query])
     similarity = cosine_similarity(query_vec, self.vectors).flatten()

     query_tokens = set(query.split())
     results = []
     for idx, score in enumerate(similarity):
        row = self.data.iloc[idx]
        uses_tokens = set(row['Uses'].split())
        overlap = len(query_tokens & uses_tokens)  # count common words

        # Hybrid score: cosine + weighted overlap
        final_score = score + 0.1 * overlap

        results.append({
            "Name": row['Name'],
            "Url": row['Url'],
            "Uses": row['Uses'],
            "Score": round(float(final_score), 4)
        })

     # Sort by hybrid score
     results = sorted(results, key=lambda x: x['Score'], reverse=True)[:top_k]
     return results

    def recommend_by_medicine(self, med_name, top_k=5):
        med_name = med_name.lower()
        match = self.data[self.data['Name'] == med_name]
        
        if match.empty:
            return f"❌ No such medicine found: {med_name}"
        
        # Take its description as the query
        med_uses = match.iloc[0]['Uses']
        print(f"ℹ️ Using '{med_name}' (uses: {med_uses}) to find alternatives...\n")
        
        return self.recommend(med_uses, top_k=top_k)




# --- Train & Save ---
data = pd.read_csv("./data/clean_med.csv")
recommender = MedicineRecommender(data)

with open("symp_recommender.pkl", "wb") as f:
    pickle.dump(recommender, f)

print("✅ Model saved as medicine_recommender.pkl")



# --- Example Usage ---
res = recommender.recommend("nose flu")
for r in res:
    if r["Score"] > 0.3:
     print(f"- {r['Name']} | {r['Uses']} : {r['Score']}")

print("________MED__________")
res = recommender.recommend_by_medicine("Panadol")
if isinstance(res, str):
    print(res)
else:
    for r in res:
        if r["Score"] > 0.3:
          print(f"- {r['Name']} | {r['Uses']} : {r['Score']}")
