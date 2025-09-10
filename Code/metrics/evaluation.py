import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
from statsmodels.stats.inter_rater import fleiss_kappa

# --- Load data ---
file_path = "overview.xlsx"
sheet_name = "german" # Change to "english" for English data and "french" for French data
df = pd.read_excel(file_path, sheet_name=sheet_name)

# --- Select relevant columns ---
h1 = df['human_de_1'][1:728].astype(int) # change here also en fr
h2 = df['human_de_2'][1:728].astype(int) # change here to en fr
llm8 = df['Llama_8B_de'][1:728].astype(int) # change here also en fr
llm70 = df['Llama_70B_de'][1:728].astype(int) # change here also en fr

# --- Helper: print pairwise agreement + kappa ---
def pairwise(name, a, b):
    print(f"{name}: "
          f"Agreement={accuracy_score(a, b):.4f}, "
          f"Cohen’s κ={cohen_kappa_score(a, b):.4f}")

print("\n=== Pairwise ===")
pairwise("human1 vs human2", h1, h2)
pairwise("human1 vs Llama-8B", h1, llm8)
pairwise("human1 vs Llama-70B", h1, llm70)
pairwise("human2 vs Llama-8B", h2, llm8)
pairwise("human2 vs Llama-70B", h2, llm70)

# --- Human pooled (majority vote if equal, drop ties) ---
pooled = np.where(h1 == h2, h1, np.nan)

print("\n=== Human pooled vs LLMs ===")
mask8 = ~np.isnan(pooled)
pairwise("human_pooled vs Llama-8B", pooled[mask8].astype(int), llm8[mask8])

mask70 = ~np.isnan(pooled)
pairwise("human_pooled vs Llama-70B", pooled[mask70].astype(int), llm70[mask70])

# --- Fleiss' Kappa (3 raters) ---
def to_fleiss_matrix(raters, n_categories):
    """Convert rater labels (n_items x n_raters) to Fleiss count matrix"""
    n_items, n_raters = raters.shape
    M = np.zeros((n_items, n_categories), dtype=int)
    for i in range(n_items):
        for r in raters[i]:
            M[i, r] += 1
    return M

n_cat = max(h1.max(), h2.max(), llm8.max(), llm70.max()) + 1

print("\n=== Fleiss’ κ ===")
raters_8B = np.vstack([h1, h2, llm8]).T
M8 = to_fleiss_matrix(raters_8B, n_cat)
print(f"humans + Llama-8B: Fleiss’ κ={fleiss_kappa(M8):.4f}")

raters_70B = np.vstack([h1, h2, llm70]).T
M70 = to_fleiss_matrix(raters_70B, n_cat)
print(f"humans + Llama-70B: Fleiss’ κ={fleiss_kappa(M70):.4f}")