import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dokulanc import config
from dokulanc.config import CLASS_LABELS
from dokulanc.document import Check
from dokulanc.pipeline import Pipeline
from dokulanc.preprocess import IMAGE_EXTS, decode_bytes
from dokulanc.report import summary_row
from dokulanc.schemas import FREE_TEXT_FIELDS, LIST_FIELDS
from dokulanc.validation import Validator

st.set_page_config(page_title="DokuLánc", page_icon="📄", layout="wide")

LEVEL_ICON = {Check.OK: "✅", Check.WARNING: "⚠️", Check.ERROR: "❌"}
LEVEL_RANK = {Check.OK: 0, Check.WARNING: 1, Check.ERROR: 2}
STEP_LABELS = {
    "betoltes": "Betöltés", "elofeldolgozas": "Előfeldolgozás", "M1_tipus": "M1 típusfelismerés",
    "M2_ocr": "M2 OCR", "M3_tipus_ellenorzes": "M3 típus-visszacsatolás", "M3_kinyeres": "M3 adatkinyerés",
    "ellenorzes": "Ellenőrzés",
}


@st.cache_resource(show_spinner="Modellek betöltése…")
def get_pipeline(ocr: str, model: str, fallback: bool, profiles: bool, threshold: float) -> Pipeline:
    pipe = Pipeline(ocr_engine=ocr, llm_model=model, llm_fallback=fallback, use_profiles=profiles, threshold=threshold)
    pipe.warmup()
    return pipe


def to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def draw_boxes(doc) -> np.ndarray:
    img = doc.image.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    thick = max(2, img.shape[1] // 600)
    for line in doc.ocr_lines:
        c = line.confidence
        color = (60, 170, 60) if c >= 0.8 else (0, 160, 255) if c >= config.LOW_OCR_CONFIDENCE else (40, 40, 220)
        x1, y1, x2, y2 = line.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, str(line.id), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thick, color, thick)
    return to_rgb(img)


def field_rows(doc) -> pd.DataFrame:
    by_field = {}
    for c in doc.checks:
        key = c.field
        cur = by_field.get(key)
        if cur is None or LEVEL_RANK[c.level] > LEVEL_RANK[cur.level]:
            by_field[key] = c
    rows = []
    for name, raw in doc.fields.items():
        if name in LIST_FIELDS:
            continue
        if isinstance(raw, dict):
            value, line = raw.get("ertek"), raw.get("sor")
        else:
            value, line = raw, None
        chk = by_field.get(name)
        status = LEVEL_ICON[chk.level] if chk else ("" if name in FREE_TEXT_FIELDS or value is None else "✅")
        rows.append({"mező": name, "érték": value or "", "forrássor": line, "állapot": status,
                     "megjegyzés": chk.message if chk and chk.level != Check.OK else ""})
    return pd.DataFrame(rows)


def apply_edits(doc, edited: pd.DataFrame):
    for _, r in edited.iterrows():
        name = r["mező"]
        raw = doc.fields.get(name)
        new = r["érték"] if r["érték"] != "" else None
        if isinstance(raw, dict):
            raw["ertek"] = new
        else:
            doc.fields[name] = new
    doc.parsed, doc.checks = Validator().validate(doc)


def show_document(doc, key: str):
    probs = doc.type_probs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dokumentumtípus", CLASS_LABELS.get(doc.doc_type or "", "—"), doc.type_source or None, delta_color="off")
    c2.metric("M1 valószínűség", f"{doc.top_probability:.0%}")
    n_err = sum(1 for c in doc.checks if c.level == Check.ERROR)
    n_warn = sum(1 for c in doc.checks if c.level == Check.WARNING)
    c3.metric("Ellenőrzés", f"{LEVEL_ICON[doc.status()]} {n_err} hiba / {n_warn} figy.")
    c4.metric("Teljes idő", f"{sum(doc.timings.values()):.1f} s")
    for e in doc.errors:
        st.error(e)

    tabs = st.tabs(["1 · Típus (M1)", "2 · Szöveg (M2)", "3 · Adatok (M3)", "4 · Ellenőrzés", "Folyamat", "JSON"])
    with tabs[0]:
        a, b = st.columns([1, 1])
        with a:
            st.caption("Eredeti kép")
            st.image(to_rgb(doc.original_image), width="stretch")
        with b:
            st.caption(f"Előfeldolgozott kép · perspektíva javítva: {'igen' if doc.preprocessing.get('perspektiva_javitva') else 'nem'}"
                       f" · dőlésszög: {doc.preprocessing.get('dolesszog', 0)}°")
            st.image(to_rgb(doc.image), width="stretch")
        if probs:
            dfp = pd.DataFrame({"típus": [CLASS_LABELS.get(k, k) for k in probs], "valószínűség": list(probs.values())})
            st.bar_chart(dfp, x="típus", y="valószínűség", horizontal=True, height=220)
        if doc.llm_type:
            st.info(f"Az M1 bizonytalan volt, az LLM a szöveg alapján: **{CLASS_LABELS.get(doc.llm_type)}** — "
                    f"{doc.preprocessing.get('llm_tipus_indoklas', '')}")
    with tabs[1]:
        st.caption(f"OCR-motor: {doc.ocr_engine} · típusfüggő profil: {doc.ocr_profile} · {len(doc.ocr_lines)} sor "
                   "· zöld ≥ 0,8, narancs ≥ 0,5, piros < 0,5 konfidencia")
        a, b = st.columns([3, 2])
        with a:
            st.image(draw_boxes(doc), width="stretch")
        with b:
            st.dataframe(pd.DataFrame([{"sor": l.id, "szöveg": l.text, "konf.": round(l.confidence, 2)} for l in doc.ocr_lines]),
                         hide_index=True, width="stretch", height=600)
    with tabs[2]:
        if not doc.fields:
            st.warning("Nincs kinyert adat.")
        else:
            st.caption("Az érték oszlop szerkeszthető; a módosítás után az ellenőrzés újrafut.")
            df = field_rows(doc)
            edited = st.data_editor(df, hide_index=True, width="stretch", disabled=["mező", "forrássor", "állapot", "megjegyzés"],
                                    key=f"edit_{key}")
            if st.button("Újraellenőrzés a javított értékekkel", key=f"recheck_{key}"):
                apply_edits(doc, edited)
                st.rerun()
            for name in LIST_FIELDS:
                items = doc.fields.get(name)
                if items:
                    st.markdown(f"**{name}**")
                    if isinstance(items[0], dict):
                        st.dataframe(pd.DataFrame(items), hide_index=True, width="stretch")
                    else:
                        st.write(", ".join(map(str, items)))
    with tabs[3]:
        if not doc.checks:
            st.info("Nincs ellenőrzési eredmény.")
        for level, fn in ((Check.ERROR, st.error), (Check.WARNING, st.warning), (Check.OK, st.success)):
            for c in [c for c in doc.checks if c.level == level]:
                fn(f"**{c.field}** — {c.message}")
    with tabs[4]:
        t = pd.DataFrame({"lépés": [STEP_LABELS.get(k, k) for k in doc.timings], "idő (s)": list(doc.timings.values())})
        st.bar_chart(t, x="lépés", y="idő (s)", horizontal=True, height=260)
        st.markdown(
            "**Lánc:** előfeldolgozás → **M1** képi típusfelismerő → a típus választja ki az OCR-profilt → **M2** OCR "
            "(sorok + pozíció + konfidencia) → ha M1 bizonytalan, **M3** a szöveg alapján újra dönt → **M3** típusfüggő "
            "JSON-kinyerés forrássorokkal → szabályalapú ellenőrzés.")
    with tabs[5]:
        st.json(doc.to_dict(), expanded=False)


def main():
    st.title("📄 DokuLánc")
    st.caption("Fotózott vagy szkennelt dokumentumok feldolgozása: típusfelismerés → OCR → LLM-alapú adatkinyerés → ellenőrzés")
    with st.sidebar:
        st.header("Beállítások")
        ocr = st.selectbox("OCR-motor (M2)", ["easyocr", "tesseract"])
        model = st.text_input("LLM (M3, Ollama)", config.LLM_MODEL)
        fallback = st.toggle("LLM-visszacsatolás bizonytalan típusnál", True)
        profiles = st.toggle("Típusfüggő OCR-előfeldolgozás", True)
        threshold = st.slider("M1 bizonytalansági küszöb", 0.3, 0.95, config.CLASSIFIER_THRESHOLD, 0.05)
        pipe = get_pipeline(ocr, model, fallback, profiles, threshold)
        st.divider()
        st.subheader("Állapot")
        import torch
        st.write(("✅ GPU: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "⚠️ Nincs GPU, CPU-n fut")
        st.write("✅ M1 modell betanítva" if config.CLASSIFIER_PATH.exists() else "❌ Nincs M1 modell")
        ok, msg = pipe.llm_status()
        st.write(f"✅ LLM: {model}" if ok else f"❌ {msg}")

    uploads = st.file_uploader("Dokumentum(ok) feltöltése", type=[e.strip(".") for e in IMAGE_EXTS] + ["pdf"],
                               accept_multiple_files=True)
    samples = sorted(p for p in config.SAMPLES_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS | {".pdf"}) if config.SAMPLES_DIR.exists() else []
    chosen = st.multiselect("…vagy válassz mintadokumentumot", [p.name for p in samples])
    inputs = [(u.name, u.getvalue()) for u in uploads or []]
    inputs += [(p.name, p.read_bytes()) for p in samples if p.name in chosen]
    if not inputs:
        st.info("Tölts fel egy képet vagy PDF-et, vagy válassz a mintákból.")
        return
    results = st.session_state.setdefault("results", {})
    settings = f"{ocr}|{model}|{fallback}|{profiles}|{threshold}"
    docs = []
    for name, data in inputs:
        key = hashlib.md5(data + settings.encode()).hexdigest()
        if key not in results:
            with st.spinner(f"Feldolgozás: {name}"):
                results[key] = pipe.process_image(decode_bytes(data, name), name)
        docs.append((key, results[key]))
    if len(docs) > 1:
        st.subheader("Összesítés")
        st.dataframe(pd.DataFrame([summary_row(d) for _, d in docs]), hide_index=True, width="stretch")
    for key, doc in docs:
        with st.expander(f"{LEVEL_ICON[doc.status()]} {doc.name} — {CLASS_LABELS.get(doc.doc_type or '', '?')}", expanded=len(docs) == 1):
            show_document(doc, key)
    all_docs = [d for _, d in docs]
    c1, c2 = st.columns(2)
    c1.download_button("Eredmények letöltése (JSON)", json.dumps([d.to_dict() for d in all_docs], ensure_ascii=False, indent=2),
                       "dokulanc_eredmenyek.json", "application/json")
    c2.download_button("Összesítő letöltése (CSV)", pd.DataFrame([summary_row(d) for d in all_docs]).to_csv(sep=";", index=False).encode("utf-8-sig"),
                       "dokulanc_eredmenyek.csv", "text/csv")


main()
