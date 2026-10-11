"""Streamlit demo: predict whether a customer will accept a personal loan.

Run from the project root:  streamlit run app/app.py
(Requires models/best_model.pkl -> run `python src/train.py` first.)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"
EDU_LABELS = {1: "Undergraduate", 2: "Graduate", 3: "Advanced / Professional"}

st.set_page_config(page_title="Personal Loan Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("models/best_model.pkl not found. Run `python src/train.py` first.")
    st.stop()

bundle = load_model()
model, features = bundle["model"], bundle["features"]

st.title("🏦 Personal Loan Acceptance Predictor")
st.caption(f"Model: **{bundle['name']}** · trained on the Thera Bank dataset (5000 customers)")

# ---------------- Inputs ----------------
# Number boxes have no min/max so out-of-range values reach Python and are validated below.
# Widgets are laid out row by row (pairs) so Tab moves left -> right, top -> bottom, ending at Predict.
LIMITS = {"Age": (18, 100), "Experience": (0, 60), "Income": (0, 300),
          "CCAvg": (0.0, 15.0), "Mortgage": (0, 700)}

r = st.columns(2)
age = r[0].number_input("Age (years)", value=35, step=1)
experience = r[1].number_input("Professional experience (years)", value=10, step=1)
r = st.columns(2)
income = r[0].number_input("Annual income ($000)", value=60, step=1)
ccavg = r[1].number_input("Avg credit-card spend / month ($000)", value=1.5, step=0.1, format="%.1f")
r = st.columns(2)
mortgage = r[0].number_input("Mortgage value ($000)", value=0, step=1)
family = r[1].selectbox("Family size", [1, 2, 3, 4], index=1)
r = st.columns(2)
education = r[0].selectbox("Education", [1, 2, 3], format_func=EDU_LABELS.get)
r = st.columns(2)
securities = r[0].checkbox("Has securities account")
cd = r[1].checkbox("Has CD (certificate of deposit) account")
r = st.columns(2)
online = r[0].checkbox("Uses internet banking")
creditcard = r[1].checkbox("Has bank's credit card")

inputs = {
    "Age": int(age), "Experience": int(experience), "Income": int(income), "Family": int(family),
    "CCAvg": round(float(ccavg), 1), "Education": int(education), "Mortgage": int(mortgage),
    "Securities Account": int(securities), "CD Account": int(cd),
    "Online": int(online), "CreditCard": int(creditcard),
}

# ---------------- Validation ----------------
errors = []
for name, (lo, hi) in LIMITS.items():
    if not lo <= inputs[name] <= hi:
        errors.append(f"{name} = {inputs[name]} is outside the allowed range {lo}–{hi}.")
if LIMITS["Age"][0] <= age and experience > age - 18:
    errors.append(f"Experience ({experience} yrs) cannot exceed Age − 18 ({age - 18} yrs).")
for e in errors:
    st.error(e)

predict = st.button("Predict", type="primary", disabled=bool(errors))
if errors:
    st.caption("Fix the highlighted values to enable Predict.")

if predict and not errors:
    row = pd.DataFrame([inputs])[features]
    st.session_state["result"] = {
        "inputs": inputs,
        "proba": float(model.predict_proba(row)[0, 1]),
        "pred": int(model.predict(row)[0]),
    }

# ---------------- Result ----------------
res = st.session_state.get("result")
if res:
    stale = res["inputs"] != inputs or bool(errors)
    if stale:
        st.info("✏️ Inputs changed. Click **Predict** to update the result.")
    else:
        pct = round(res["proba"] * 100, 1)
        if res["pred"] == 1:
            st.success(f"✅ Likely to **ACCEPT** the personal loan — probability {pct:.1f}%")
        else:
            st.warning(f"❌ Unlikely to accept the personal loan — probability {pct:.1f}%")
        st.progress(min(100, max(0, int(round(pct)))), text=f"{pct:.1f}%")

        shown = dict(res["inputs"])
        shown["Education"] = f'{shown["Education"]} ({EDU_LABELS[shown["Education"]]})'
        summary = pd.DataFrame({"Feature": list(shown.keys()),
                                "Value used": [str(v) for v in shown.values()]})
        st.markdown("**Values used for this prediction**")
        st.dataframe(summary, hide_index=True, height=35 * (len(summary) + 1) + 3)

st.divider()
st.caption("Try: Income 180, Education Graduate, Family 3, CCAvg 4.5, CD account → high probability. "
           "Income 40, Undergraduate, Family 1 → low probability. "
           "Education effect: Income 120, Family 1, CCAvg 2.5 → Undergraduate ≈ 0% vs Graduate ≈ 100%.")
