"""
ChestX6 - Live diagnosis viewer.

Lance :
    python3 -m streamlit run streamlit/app.py
"""
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --- Chemins ---
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_VAL       = PROJECT_ROOT / "Data"       / "val"
OUTPUT_VAL     = PROJECT_ROOT / "output"     / "val"
VAL_PARQUET    = PROJECT_ROOT / "results"    / "val.parquet"
LABELS_JSON    = PROJECT_ROOT / "output"     / "chestx6_labels.json"
CHECKPOINT_VAL = PROJECT_ROOT / "checkpoint" / "val"
RESULTS_JSON   = PROJECT_ROOT / "output"     / "resultats_test.json"
CONSUMER_LOG   = PROJECT_ROOT / "consumer.log"

BATCH_SIZE   = 5   # images copiees par batch
INTERVAL_SEC = 5   # secondes entre chaque batch

# --- Page ---
st.set_page_config(page_title="ChestX6 - Live", page_icon="🫁", layout="wide")

# --- Classes ---
try:
    CLASSES = json.loads(LABELS_JSON.read_text())
except Exception:
    CLASSES = ["Covid-19", "Emphysema", "Normal", "Pneumonia-Bacterial"]

# --- Session state ---
st.session_state.setdefault("streaming",           False)
st.session_state.setdefault("sent_paths",          set())
st.session_state.setdefault("consumer_proc",       None)
st.session_state.setdefault("consumer_start_time", None)

# ==================== Gestion consumer subprocess ====================

def is_consumer_running() -> bool:
    p = st.session_state.consumer_proc
    return p is not None and p.poll() is None


def start_consumer():
    if is_consumer_running():
        return
    env = os.environ.copy()
    env["JAVA_OPTS"] = "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    env["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    env["PATH"]      = f"/opt/homebrew/bin:{env['JAVA_HOME']}/bin:{env.get('PATH','')}"

    log = open(CONSUMER_LOG, "w")
    p = subprocess.Popen(
        'sbt "runMain ImageConsumer"',
        shell=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,   # groupe process independant (pour tuer proprement)
    )
    st.session_state.consumer_proc       = p
    st.session_state.consumer_start_time = time.time()


def stop_consumer():
    p = st.session_state.consumer_proc
    if p and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    st.session_state.consumer_proc       = None
    st.session_state.consumer_start_time = None


def consumer_status_text() -> str:
    p = st.session_state.consumer_proc
    if p is None:
        return "🔴 arrêté"
    rc = p.poll()
    if rc is not None:
        return f"❌ crashé (exit {rc})"
    elapsed = int(time.time() - (st.session_state.consumer_start_time or time.time()))
    if elapsed < 45:
        return f"🟡 démarrage… {elapsed}s"
    return "🟢 en cours"


# ==================== HEADER ====================

st.title("🫁 ChestX6 — Diagnostic radiologique en temps réel")
st.caption("Flux : Data/val → output/val → consumer Scala (ONNX predict) → val.parquet → cette page")

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("🎛️ Pipeline")

    running = st.session_state.streaming or is_consumer_running()

    if not running:
        if st.button("▶️ Lancer TOUT (consumer + streaming)",
                     type="primary", use_container_width=True):
            # 1. Nettoyer l'ancien val
            for p in (OUTPUT_VAL, VAL_PARQUET, CHECKPOINT_VAL):
                if p.exists():
                    shutil.rmtree(p)
            OUTPUT_VAL.mkdir(parents=True, exist_ok=True)
            st.session_state.sent_paths = set()

            # 2. Lancer le consumer
            start_consumer()

            # 3. Activer le streaming (copie val)
            st.session_state.streaming = True
            st.rerun()
    else:
        if st.button("⏹️ Arrêter TOUT", type="primary", use_container_width=True):
            st.session_state.streaming = False
            stop_consumer()
            st.rerun()

    st.markdown(f"**Consumer** : {consumer_status_text()}")
    st.markdown(f"**Streaming val** : {'🟢 ON' if st.session_state.streaming else '🔴 OFF'}")
    st.markdown(f"**Envoyées** : {len(st.session_state.sent_paths)}")

    st.divider()

    # --- UPLOAD utilisateur ---
    st.subheader("📤 Importer des photos")
    uploaded = st.file_uploader(
        "1 à 5 images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="uploader",
    )
    upload_class = st.selectbox("Classe (dossier de dépôt)", CLASSES, key="upload_class")

    if uploaded:
        n = min(len(uploaded), 5)
        if st.button(f"📩 Envoyer {n} image(s)", use_container_width=True):
            target_dir = OUTPUT_VAL / upload_class
            target_dir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time())
            for i, f in enumerate(uploaded[:5]):
                dest = target_dir / f"user_{stamp}_{i}_{f.name}"
                dest.write_bytes(f.read())
            st.success(f"{n} image(s) envoyée(s) dans output/val/{upload_class}/")

# ==================== STREAMING (copie Data/val -> output/val) ====================

if st.session_state.streaming and is_consumer_running():
    batch = []
    if DATA_VAL.exists():
        for cls_dir in sorted(DATA_VAL.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img in sorted(cls_dir.iterdir()):
                if (img.is_file()
                    and img.suffix.lower() in (".jpg", ".jpeg", ".png")
                    and str(img) not in st.session_state.sent_paths):
                    batch.append((img, cls_dir.name))
                    if len(batch) >= BATCH_SIZE:
                        break
            if len(batch) >= BATCH_SIZE:
                break

    if batch:
        for img_path, cls in batch:
            dest_dir = OUTPUT_VAL / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dest_dir / img_path.name)
            st.session_state.sent_paths.add(str(img_path))
    # Note : on ne desactive PAS streaming si batch vide,
    # comme ca l'user peut uploader des photos qui declenchent quand meme.

# ==================== LECTURE parquet val ====================

try:
    # Lit chaque part-file dans l'ordre de creation (mtime) pour preserver
    # l'ordre d'arrivee des batches -> le vrai "newest last".
    parts = sorted(
        VAL_PARQUET.glob("part-*.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    if parts:
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df = pd.read_parquet(VAL_PARQUET)
    if "prediction_name" not in df.columns:
        df["prediction_name"] = "N/A"
except Exception:
    df = pd.DataFrame(columns=["path", "label", "prediction_name"])

# ==================== KPIs ====================

c1, c2, c3, c4 = st.columns(4)
c1.metric("📸 Images traitées", len(df))

if len(df) > 0:
    ok  = int((df["label"] == df["prediction_name"]).sum())
    acc = ok / len(df)
    c2.metric("🎯 Accuracy live", f"{acc:.1%}", delta=f"{ok}/{len(df)} OK")
    last = df.tail(1).iloc[0]
    c3.metric("🩺 Dernier diag.", str(last["prediction_name"]))
else:
    c2.metric("🎯 Accuracy live", "—")
    c3.metric("🩺 Dernier diag.", "—")

c4.metric("⚡ Consumer", consumer_status_text())

st.divider()

# ==================== TIMELINE + BAR CHART ====================

if len(df) > 0:
    # --- Marquage source (upload utilisateur vs stream) ---
    df["fichier"] = df["path"].str.split("/").str[-1]
    df["source"]  = np.where(df["fichier"].str.startswith("user_"), "👤 upload", "📸 stream")

    # --- Uploads utilisateur en HAUT ---
    user_df = df[df["source"] == "👤 upload"].copy()
    if len(user_df) > 0:
        st.success(f"👤 **{len(user_df)} photo(s) importée(s)** — voir ci-dessous")
        u = user_df.tail(10).iloc[::-1].copy()
        u["✓"] = np.where(u["label"] == u["prediction_name"], "✅", "❌")
        u["nom original"] = u["fichier"].str.replace(r"^user_\d+_\d+_", "", regex=True)
        st.dataframe(
            u[["nom original", "label", "prediction_name", "✓"]].rename(columns={
                "label": "Classe déposée", "prediction_name": "Prédit",
            }),
            use_container_width=True, hide_index=True,
        )
        st.divider()

    st.subheader("📈 Arrivée des batches en temps réel")
    st.caption("Chaque courbe = une classe. À chaque batch de 5 images qui arrive, "
               "les courbes montent selon les prédictions.")

    # Numero de batch : batch 1 = images 1-5, batch 2 = images 6-10, etc.
    df_ln = df.copy().reset_index(drop=True)
    df_ln["Batch #"] = (df_ln.index // 5) + 1

    # Compteur cumulatif par classe a la fin de chaque batch
    max_batch = int(df_ln["Batch #"].max())
    rows = []
    for b in range(1, max_batch + 1):
        sub = df_ln[df_ln["Batch #"] <= b]
        for cls in CLASSES:
            rows.append({
                "Batch #": b,
                "Classe": cls,
                "Nb cumulé": int((sub["prediction_name"] == cls).sum()),
            })
    tl = pd.DataFrame(rows)

    chart = (alt.Chart(tl)
             .mark_line(strokeWidth=3, point=alt.OverlayMarkDef(size=60, filled=True))
             .encode(
                 x=alt.X("Batch #:Q",  title="Batch # (5 images / batch)"),
                 y=alt.Y("Nb cumulé:Q", title="Nombre cumulé d'images prédites"),
                 color=alt.Color("Classe:N",
                                 legend=alt.Legend(title="Classe prédite")),
                 tooltip=[
                     alt.Tooltip("Batch #:Q"),
                     alt.Tooltip("Classe:N"),
                     alt.Tooltip("Nb cumulé:Q"),
                 ],
             )
             .properties(height=340))
    st.altair_chart(chart, use_container_width=True)

    # --- Bar chart : contenu de chaque batch ---
    st.subheader("📊 Contenu de chaque batch")
    st.caption("Chaque barre = 1 batch (max 5 images). Les couleurs empilées = les classes prédites dans ce batch.")

    by_batch = (df_ln.groupby(["Batch #", "prediction_name"])
                     .size()
                     .reset_index(name="Nb images"))

    bar = (alt.Chart(by_batch)
           .mark_bar()
           .encode(
               x=alt.X("Batch #:O",     title="Batch # (ordre d'arrivée)"),
               y=alt.Y("Nb images:Q",   title="Images dans le batch (max 5)"),
               color=alt.Color("prediction_name:N",
                               legend=alt.Legend(title="Classe prédite")),
               tooltip=[
                   alt.Tooltip("Batch #:O"),
                   alt.Tooltip("prediction_name:N", title="Classe"),
                   alt.Tooltip("Nb images:Q"),
               ],
           )
           .properties(height=280))
    st.altair_chart(bar, use_container_width=True)

    st.subheader("🔬 20 dernières prédictions")
    st.caption("Les 20 dernières arrivées, du plus récent (en haut) au plus ancien. Stream ou upload.")

    recent = df.tail(20).iloc[::-1].copy()   # dernier en haut, sans distinction
    recent["✓"] = np.where(recent["label"] == recent["prediction_name"], "✅", "❌")
    st.dataframe(
        recent[["source", "fichier", "label", "prediction_name", "✓"]].rename(columns={
            "label": "Réel", "prediction_name": "Prédit",
        }),
        use_container_width=True, hide_index=True, height=440,
    )

    st.subheader("📋 Détail par classe")
    per_class = []
    for cls in CLASSES:
        actuel = df[df["label"] == cls]
        if len(actuel) > 0:
            corrects = int((actuel["prediction_name"] == cls).sum())
            per_class.append({
                "classe": cls,
                "total réel": len(actuel),
                "correctement prédit": corrects,
                "accuracy": f"{corrects/len(actuel):.1%}",
            })
    if per_class:
        st.dataframe(pd.DataFrame(per_class), use_container_width=True, hide_index=True)
else:
    if st.session_state.streaming or is_consumer_running():
        st.info("⏳ Démarrage du consumer et arrivée des premières prédictions… (30-45s pour booter Spark)")
    else:
        st.info("👉 Clique **▶️ Lancer TOUT** dans la sidebar pour démarrer le pipeline.")

# ==================== AUTO-REFRESH ====================

if st.session_state.streaming or is_consumer_running():
    time.sleep(INTERVAL_SEC)
    st.rerun()
