"""
ChestX6 - Application Streamlit (page unique, 2 modes).

  Mode 1 - Diagnostic medecin : upload de 1 a 5 radios -> diagnostic + confiance.
  Mode 2 - Validation          : accuracy en temps reel sur les images val.

IMPORTANT : cette appli NE lance PAS le consumer. Lance-le toi-meme dans un terminal :
    sbt "runMain ImageConsumer"

Lancer l'appli :
    python -m streamlit run streamlit/app.py
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ==================== Chemins ====================
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
UPLOAD_DIR     = PROJECT_ROOT / "output"  / "upload"
STAGING_DIR    = PROJECT_ROOT / "output"  / "_staging"
OUTPUT_VAL     = PROJECT_ROOT / "output"  / "val"
UPLOAD_PARQUET = PROJECT_ROOT / "results" / "upload.parquet"
VAL_PARQUET    = PROJECT_ROOT / "results" / "val.parquet"

PREDICTION_TIMEOUT = 90
POLL_INTERVAL      = 2
REFRESH_SEC        = 4
MAX_UPLOAD         = 5     # nombre max d'images uploadees d'un coup

# ==================== Libelles medicaux (6 classes) ====================
DIAG_LABELS = {
    "Normal":              ("Poumons sains",         "#16a34a"),
    "Covid-19":            ("COVID-19",              "#dc2626"),
    "Pneumonia-Bacterial": ("Pneumonie bacterienne", "#ea580c"),
    "Pneumonia-Viral":     ("Pneumonie virale",      "#f59e0b"),
    "Emphysema":           ("Emphyseme",             "#ca8a04"),
    "Tuberculosis":        ("Tuberculose",           "#7c3aed"),
}
CLASSES = list(DIAG_LABELS.keys())

def friendly(name: str):
    return DIAG_LABELS.get(name, (name, "#334155"))

# ==================== Lecture parquet ====================
def read_parquet_dir(folder, columns=None) -> pd.DataFrame:
    try:
        parts = sorted(
            (f for f in folder.glob("part-*.parquet") if f.stat().st_size > 0),
            key=lambda p: p.stat().st_mtime,
        )
        if parts:
            return pd.concat([pd.read_parquet(f, columns=columns) for f in parts],
                             ignore_index=True)
        return pd.read_parquet(folder, columns=columns)
    except Exception:
        return pd.DataFrame(columns=columns or ["path", "prediction_name", "score"])

# ==================== Page ====================
st.set_page_config(page_title="ChestX6", page_icon="🫁", layout="wide")
st.session_state.setdefault("history", [])   # liste de dicts : nom, diagnostic, confiance, heure

with st.sidebar:
    st.header("🫁 ChestX6")
    mode = st.radio("Mode", ["🩺 Diagnostic medecin", "📊 Validation"])

# ============================================================
# MODE 1 — DIAGNOSTIC MEDECIN
# ============================================================
if mode.startswith("🩺"):
    st.title("🩺 Aide au diagnostic")
    st.caption("Analyse assistee de radiographies thoraciques. "
               "Outil d'aide a la decision — ne remplace pas l'avis d'un medecin.")

    def wait_for_prediction(fname, timeout=PREDICTION_TIMEOUT):
        """Renvoie (prediction_name, score) ou (None, None) si timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            df = read_parquet_dir(UPLOAD_PARQUET, columns=["path", "prediction_name", "score"])
            if len(df) and "path" in df.columns:
                hit = df[df["path"].astype(str).str.contains(fname, regex=False)]
                if len(hit):
                    row = hit.iloc[-1]
                    return str(row["prediction_name"]), float(row["score"])
            time.sleep(POLL_INTERVAL)
        return None, None

    def deposit_image(uploaded_file):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"user_{int(time.time()*1000)}_{uploaded_file.name}"
        tmp = STAGING_DIR / fname
        tmp.write_bytes(uploaded_file.getbuffer())
        os.replace(tmp, UPLOAD_DIR / fname)
        return fname

    uploaded_files = st.file_uploader(
        f"Radiographies thoraciques (1 a {MAX_UPLOAD})",
        type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        uploaded_files = uploaded_files[:MAX_UPLOAD]
        cols = st.columns(len(uploaded_files))
        for col, f in zip(cols, uploaded_files):
            col.image(f, caption=f.name, use_container_width=True)

    if st.button("🔬 Predire", type="primary", disabled=(not uploaded_files)):
        # 1. Deposer toutes les images (le consumer les prend dans le meme batch)
        pending = [(f.name, deposit_image(f)) for f in uploaded_files]

        # 2. Recuperer chaque diagnostic + confiance
        results = []
        with st.spinner(f"Analyse de {len(pending)} radiographie(s)…"):
            for original_name, fname in pending:
                diag, score = wait_for_prediction(fname)
                results.append((original_name, diag, score))

        # 3. Afficher
        for original_name, diag, score in results:
            if diag is None:
                st.error(f"{original_name} : aucun resultat (le consumer tourne-t-il ?).")
            elif diag == "unknown":
                st.warning(f"{original_name} : modele indisponible.")
            else:
                label, color = friendly(diag)
                pct = f"{score*100:.0f}%"
                st.markdown(f"""
                    <div style="border:2px solid {color}; border-radius:12px;
                                padding:16px 24px; margin:8px 0; max-width:560px;
                                display:flex; align-items:center; justify-content:space-between;">
                        <div>
                            <div style="font-size:13px; color:#64748b;">{original_name}</div>
                            <div style="font-size:26px; font-weight:700; color:{color};">{label}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:12px; color:#64748b;">CONFIANCE</div>
                            <div style="font-size:26px; font-weight:700; color:{color};">{pct}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                st.session_state.history.insert(0, {
                    "Radiographie": original_name, "Diagnostic": label,
                    "Confiance": round(score, 4), "Heure": time.strftime("%H:%M:%S")})

    # ---- Metriques de session ----
    hist = st.session_state.history
    if hist:
        st.divider()
        st.subheader("📊 Metriques de la session")
        h = pd.DataFrame(hist)

        k1, k2 = st.columns(2)
        k1.metric("Analyses effectuees", len(h))
        k2.metric("Confiance moyenne", f"{h['Confiance'].mean()*100:.0f}%")

        colA, colB = st.columns([1, 1])
        with colA:
            st.caption("Repartition des diagnostics")
            repartition = h["Diagnostic"].value_counts()
            st.bar_chart(repartition, height=260)
        with colB:
            st.caption("Historique")
            show = h.copy()
            show["Confiance"] = (show["Confiance"] * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(show, use_container_width=True, hide_index=True, height=260)

# ============================================================
# MODE 2 — DASHBOARD VALIDATION
# ============================================================
else:
    st.title("📊 Validation du modele — accuracy en temps reel")
    st.caption("Le consumer predit les images de validation (vraie classe connue). "
               "Lance-le dans un terminal, puis regarde l'accuracy monter ici.")

    auto = st.sidebar.checkbox("🔄 Rafraichir automatiquement", value=True)

    df = read_parquet_dir(VAL_PARQUET, columns=["path", "label", "prediction_name", "score"])
    if len(df):
        df = df[df["prediction_name"] != "unknown"].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📸 Images traitees", len(df))
    if len(df) > 0:
        ok  = int((df["label"] == df["prediction_name"]).sum())
        acc = ok / len(df)
        c2.metric("🎯 Accuracy live", f"{acc:.1%}", delta=f"{ok}/{len(df)} corrects")
        c3.metric("🩺 Derniere prediction", str(df.tail(1).iloc[0]["prediction_name"]))
        if "score" in df.columns:
            c4.metric("📈 Confiance moyenne", f"{df['score'].mean()*100:.0f}%")
    else:
        c2.metric("🎯 Accuracy live", "—")
        c3.metric("🩺 Derniere prediction", "—")
        c4.metric("📈 Confiance moyenne", "—")

    st.divider()

    if len(df) > 0:
        st.subheader("📈 Accuracy cumulee")
        d = df.reset_index(drop=True)
        d["correct"] = (d["label"] == d["prediction_name"]).astype(int)
        d["Accuracy cumulee"] = d["correct"].cumsum() / (d.index + 1)
        d["Image #"] = d.index + 1
        st.line_chart(d.set_index("Image #")[["Accuracy cumulee"]], height=300)

        colA, colB = st.columns(2)
        with colA:
            st.subheader("📋 Detail par classe")
            rows = []
            for cls in CLASSES:
                sub = df[df["label"] == cls]
                if len(sub) > 0:
                    corr = int((sub["prediction_name"] == cls).sum())
                    rows.append({"Classe": cls, "Total": len(sub),
                                 "Corrects": corr, "Accuracy": f"{corr/len(sub):.1%}"})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with colB:
            st.subheader("🔬 20 dernieres predictions")
            recent = df.tail(20).iloc[::-1].copy()
            recent["✓"] = np.where(recent["label"] == recent["prediction_name"], "✅", "❌")
            recent["fichier"] = recent["path"].str.split(r"[\\/]").str[-1]
            st.dataframe(recent[["fichier", "label", "prediction_name", "✓"]].rename(
                columns={"label": "Reel", "prediction_name": "Predit"}),
                use_container_width=True, hide_index=True, height=360)
    else:
        st.info("⏳ Aucune prediction pour l'instant. Lance le consumer dans un terminal "
                "(et verifie que output/val contient des images).")

    if auto:
        time.sleep(REFRESH_SEC)
        st.rerun()
