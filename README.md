# DokuLánc – többmodelles dokumentumfeldolgozó lánc

Fotózott vagy szkennelt magyar dokumentumok (számla, nyugta, levél, önéletrajz, űrlap) automatikus
feldolgozása három egymásra épülő AI-modellel, majd szabályalapú adatellenőrzéssel.

```
kép / PDF
  → előfeldolgozás (perspektíva- és dőlésjavítás)
  → M1  saját tanítású képi típusfelismerő (CNN, transfer learning)
        a típus választja ki az OCR-profilt és a kinyerési sémát
  → M2  OCR (EasyOCR, összevetésként Tesseract) – sorok + pozíció + konfidencia
  → M3  helyi LLM (Ollama, qwen2.5:7b) – típusfüggő JSON forrássorokkal
        ha M1 bizonytalan, M3 a szöveg alapján is dönt a típusról (visszacsatolás)
  → ellenőrzés: összegek, adószám, IBAN, TAJ, adóazonosító, dátumok, hallucináció-szűrés
  → Streamlit felület / CLI, JSON és CSV export
```

## Telepítés

Követelmények: Python 3.11+, NVIDIA GPU (CPU-n is fut, de lassan), [Ollama](https://ollama.com),
opcionálisan [Tesseract 5](https://github.com/UB-Mannheim/tesseract/wiki).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
ollama pull qwen2.5:7b
```

A Tesseract magyar modellje (`hun.traineddata`, tessdata_best) a `models/tessdata` mappában van.

## Futtatás

```bash
python -m dokulanc status                                  # környezet ellenőrzése
streamlit run app/streamlit_app.py                         # webes felület
python -m dokulanc process data/samples -v --json out/eredmeny.json --csv out/eredmeny.csv
python -m dokulanc classify data/samples                   # csak M1
python -m dokulanc ocr data/samples/nyugta_foto.jpg --profile nyugta   # csak M2
```

Kapcsolók: `--ocr tesseract`, `--model <ollama-modell>`, `--no-llm`, `--no-fallback`, `--no-profiles`.

## Adatok és tanítás

```bash
python training/generate_documents.py --clean   # szintetikus dokumentumok ismert helyes adatokkal
python training/download_rvl.py                 # RVL-CDIP részhalmaz (Hugging Face)
python training/prepare_dataset.py              # előfeldolgozás, egységes méret, manifest
python training/train_classifier.py             # M1: 3 architektúra tanítása, a legjobb mentése
python training/make_samples.py                 # demó dokumentumok a data/samples mappába
```

Saját fotók a `data/real/<típus>/` mappába tehetők (`szamla`, `nyugta`, `level`, `oneletrajz`, `urlap`);
a `prepare_dataset.py` ezeket külön, csak tesztelésre használt halmazba teszi.

## Kiértékelés

```bash
python evaluation/eval_classifier.py      # M1: accuracy, makró F1, konfúziós mátrix, forrásonként
python evaluation/eval_ocr.py             # M2: CER, WER, szó-felidézés; motorok és profilok
python evaluation/eval_extraction.py      # M3: mezőpontosság ideális vs. OCR-szövegen (hibaterjedés)
python evaluation/eval_pipeline.py        # teljes lánc, elrontott dokumentumok felismerése, futási idők
pytest                                    # egységtesztek (az LLM-tesztek futó Ollamát igényelnek)
```

Az eredmények a `reports/results` (CSV), az ábrák a `reports/figures` mappába kerülnek.

## Könyvtárszerkezet

```
src/dokulanc/
  config.py        útvonalak, osztályok, küszöbök, modellnevek
  document.py      Document, OcrLine, Check adatmodell
  preprocess.py    betöltés (kép, PDF), perspektíva- és dőlésjavítás, típusfüggő OCR-profilok
  classifier.py    M1 modell felépítése, betöltése, predikció
  ocr.py           M2 motorok (EasyOCR, Tesseract), sorokba és hasábokba rendezés, oszlopjelölés
  schemas.py       típusonkénti pydantic sémák az LLM strukturált kimenetéhez
  prompts.py       rendszer- és típusfüggő promptok
  extractor.py     Ollama-kliens, M3 adatkinyerés és típus-visszacsatolás, újrapróbálás
  validation.py    ellenőrző réteg
  repair.py        javító lépések: tételsor-rekonstrukció, címke alapú végösszeg, eladó–vevő csere
  identifiers.py   adószám, IBAN, bankszámla, TAJ, adóazonosító jel ellenőrzése és generálása
  normalize.py     összeg- és dátumértelmezés, szövegnormalizálás
  synth.py         szintetikus dokumentumgenerátor, fotó- és szkenneffektus
  metrics.py       CER, WER, mezőszintű összehasonlítás
  pipeline.py      a lánc összekötése, időmérés
  report.py        JSON / CSV export
  cli.py           parancssori felület
app/streamlit_app.py
training/          adatgenerálás, letöltés, előkészítés, M1 tanítás
evaluation/        kiértékelő szkriptek
tests/             pytest tesztek
data/ models/ reports/
```

## Felhasznált források

- RVL-CDIP (Harley et al., 2015), részhalmaz: `jordyvl/rvl_cdip_100_examples_per_class` (Hugging Face)
- torchvision ImageNet-előtanított súlyok: MobileNetV3-Large, EfficientNet-B0, ResNet-18
- EasyOCR (JaidedAI), Tesseract 5 + `tessdata_best` magyar modell
- Qwen2.5-7B-Instruct (Alibaba), Ollama futtatókörnyezet
- Könyvtárak: PyTorch, OpenCV, PyMuPDF, Faker, pydantic, rapidfuzz, scikit-learn, pandas, matplotlib, Streamlit, pytest
