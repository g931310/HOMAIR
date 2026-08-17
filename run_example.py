from pathlib import Path
import pandas as pd

from src.public_core import prepare_analysis_ready
from src.oof_models import classification_oof
from src.nested_threshold import select_thresholds

DATA = Path(__file__).parent / "data" / "synthetic_example.csv"

df = pd.read_csv(DATA)
df, tiers = prepare_analysis_ready(df)

print(f"Synthetic rows: {len(df)}")
print(f"Synthetic HOMA-IR >=2.5: {df['IR_2p5'].sum()} ({df['IR_2p5'].mean():.1%})")
print("NOTE: these are synthetic demonstration data, not study results.\n")

# The quick example runs only the full-tier CatBoost OOF classifier.
# The public modules also expose all comparator models, regression,
# nested threshold validation, CatBoost tuning sensitivity, statistics and SHAP.
full = tiers["full"]
cls = classification_oof(df, full["numeric"], full["categorical"], models=("CatBoost",))
result = cls["CatBoost"]
prob = result["probabilities"]
thresholds = select_thresholds(df["IR_2p5"].to_numpy(int), prob)

print("Synthetic CatBoost five-fold OOF classification at p=0.50:")
print({k: v for k, v in result.items() if k != "probabilities"})
print("\nThresholds selected directly on the synthetic OOF example (demonstration only):")
print(thresholds)
print("\nFor the manuscript procedure, use nested_threshold_validation() so threshold selection occurs only in inner CV.")
