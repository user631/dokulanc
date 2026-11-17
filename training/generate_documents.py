import argparse
import random
import shutil
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from dokulanc import config
from dokulanc.synth import Generator, save_rendered

SPLITS = {"train": 0, "val": 1, "test": 2}


def variant_for(rng: random.Random) -> str:
    r = rng.random()
    return "photo" if r < 0.4 else "scan" if r < 0.8 else "clean"


def job(args):
    split, doc_type, start, count, corrupt_rate, seed = args
    gen = Generator(seed)
    rng = random.Random(seed + 1)
    out = config.SYNTH_DIR / split / doc_type
    for i in range(start, start + count):
        corrupt = rng.random() < corrupt_rate
        r = gen.generate(doc_type, corrupt=corrupt)
        save_rendered(r, out, f"{doc_type}_{i:05d}", variant_for(rng), rng)
    return count


def main():
    ap = argparse.ArgumentParser(description="Szintetikus magyar dokumentumok generálása ismert helyes adatokkal")
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--corrupt", type=float, default=0.15, help="Szándékosan elrontott dokumentumok aránya")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    if args.clean and config.SYNTH_DIR.exists():
        shutil.rmtree(config.SYNTH_DIR)
    counts = {"train": args.train, "val": args.val, "test": args.test}
    jobs = []
    chunk = 50
    for split, n in counts.items():
        for t_idx, doc_type in enumerate(config.CLASSES):
            for start in range(0, n, chunk):
                seed = args.seed * 1_000_003 + SPLITS[split] * 100_000 + t_idx * 10_000 + start
                jobs.append((split, doc_type, start, min(chunk, n - start), args.corrupt, seed))
    total = sum(j[3] for j in jobs)
    with ProcessPoolExecutor(args.workers) as ex, tqdm(total=total, desc="generálás") as bar:
        for n in ex.map(job, jobs):
            bar.update(n)
    print(f"Kész: {total} dokumentum -> {config.SYNTH_DIR}")


if __name__ == "__main__":
    main()
