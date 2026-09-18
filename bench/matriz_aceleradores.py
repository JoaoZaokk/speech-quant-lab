"""Matriz completa de aceleradores do treino do Fish: kernel x compile x batch.

POR QUE MATRIZ E NAO ESCADA
---------------------------
Os ganhos NAO multiplicam, e ha' um motivo mecanico: `with sdpa_kernel(...)` e'
um context manager Python, e o `torch.compile` QUEBRA O GRAFO quando encontra
um. Sao 36 camadas — 36 quebras. Entao "cuDNN sozinho ganha" e "compile sozinho
ganha" nao implica "cuDNN + compile ganha mais". Pode perder dos dois.

Por isso cada celula e' um processo proprio, com a mesma seed e o mesmo dado, e
a tabela sai inteira antes de qualquer conclusao.

    python scripts/matriz_aceleradores.py
    python scripts/matriz_aceleradores.py --attn math,cudnn
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable   # o mesmo interpretador que roda este script


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attn", default="math,effic,cudnn,flash")
    ap.add_argument("--batch", default="1,4")
    ap.add_argument("--tokens", type=int, default=160,
                    help="comprimento real das amostras do pack (mediana 145)")
    ap.add_argument("--passos", type=int, default=8)
    cli = ap.parse_args()

    linhas = []
    for attn in [a.strip() for a in cli.attn.split(",") if a.strip()]:
        for batch in [int(b) for b in cli.batch.split(",")]:
            env = {**os.environ, "FISH_ATTN": attn}
            cmd = [PY, str(ROOT / "scripts" / "bench_treino_fish.py"),
                   "--lista", "base,compile", "--passos", str(cli.passos),
                   "--aquecer", "4", "--max-length", str(cli.tokens),
                   "--batch", str(batch)]
            print(f"  rodando attn={attn} batch={batch} ...", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env)
            vals = re.findall(r"([\d.]+) s/passo", r.stdout)
            oom = "OutOfMemory" in r.stdout + r.stderr
            for i, comp in enumerate(("nao", "sim")):
                if i < len(vals):
                    s = float(vals[i])
                    linhas.append({"attn": attn, "compile": comp, "batch": batch,
                                   "s_passo": s, "amostras_s": batch / s})
                elif oom:
                    linhas.append({"attn": attn, "compile": comp, "batch": batch,
                                   "s_passo": None, "amostras_s": None})

    if not linhas:
        print("nenhuma medicao saiu — ver o stderr do bench")
        return 1

    ref = next((x["amostras_s"] for x in linhas
                if x["attn"] == "math" and x["compile"] == "nao" and x["batch"] == 1),
               linhas[0]["amostras_s"])

    print(f"\n{'attn':<8} {'compile':<8} {'batch':>5} {'s/passo':>9} "
          f"{'amostras/s':>11} {'ganho':>7}")
    print("-" * 54)
    for x in linhas:
        if x["s_passo"] is None:
            print(f"{x['attn']:<8} {x['compile']:<8} {x['batch']:>5} {'OOM':>9}")
            continue
        print(f"{x['attn']:<8} {x['compile']:<8} {x['batch']:>5} "
              f"{x['s_passo']:>9.3f} {x['amostras_s']:>11.2f} "
              f"{x['amostras_s']/ref:>6.2f}x")

    melhor = max((x for x in linhas if x["amostras_s"]), key=lambda x: x["amostras_s"])
    print(f"\nmelhor: attn={melhor['attn']} compile={melhor['compile']} "
          f"batch={melhor['batch']}  {melhor['amostras_s']/ref:.2f}x")

    out = ROOT / "reports" / "matriz_aceleradores.json"
    out.write_text(json.dumps(linhas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
