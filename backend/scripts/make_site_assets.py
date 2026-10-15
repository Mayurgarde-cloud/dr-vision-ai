"""Create the artwork used by the website (hero fundus + five severity tiles).

    python -m scripts.make_site_assets     (writes into ../frontend/public/samples)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.synthetic import make_fundus  # noqa: E402

OUT = Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "samples"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_fundus(2, seed=11, size=900, transparent=True).save(OUT / "hero.png")
    for sev in range(5):
        make_fundus(sev, seed=3, size=360).save(OUT / f"class_{sev}.jpg", quality=90)
    make_fundus(2, seed=3, size=240).save(OUT / "thumb.jpg", quality=90)
    print("Wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
