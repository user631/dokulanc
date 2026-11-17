import json
import random
import shutil

from dokulanc import config
from dokulanc.synth import Generator, photo_effect, scan_effect


def main():
    out = config.SAMPLES_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    gen = Generator(2026)
    rng = random.Random(2026)
    truth = {}
    for t in config.CLASSES:
        r = gen.generate(t)
        photo_effect(r.image, rng).save(out / f"{t}_foto.jpg", quality=90)
        truth[f"{t}_foto.jpg"] = {"tipus": t, "mezok": r.fields, "hibak": r.errors}
        r = gen.generate(t)
        scan_effect(r.image, rng).save(out / f"{t}_szkennelt.png")
        truth[f"{t}_szkennelt.png"] = {"tipus": t, "mezok": r.fields, "hibak": r.errors}
    for t in ("szamla", "nyugta", "urlap"):
        r = gen.generate(t, corrupt=True)
        photo_effect(r.image, rng).save(out / f"{t}_hibas_foto.jpg", quality=90)
        truth[f"{t}_hibas_foto.jpg"] = {"tipus": t, "mezok": r.fields, "hibak": r.errors}
    r = gen.generate("szamla")
    r.image.convert("RGB").save(out / "szamla_digitalis.pdf", "PDF", resolution=150)
    truth["szamla_digitalis.pdf"] = {"tipus": "szamla", "mezok": r.fields, "hibak": r.errors}
    (out / "helyes_ertekek.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(truth)} mintadokumentum -> {out}")


if __name__ == "__main__":
    main()
