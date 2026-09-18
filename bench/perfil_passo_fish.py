"""Onde o tempo do passo de treino do Fish vai DE VERDADE, kernel a kernel.

POR QUE ISTO E PRECISO
----------------------
MEDIDO em 2026-09-18: o GEMM INT8 do comfy-kitchen e' 1,85x a 2,58x mais rapido
que o bf16 nas formas do s2-pro (`scripts/bench_int8_kernel.py`). Trocados os
201 lineares do modelo, o passo de treino nao mudou: 0,266 contra 0,269 s.

As duas explicacoes sao incompativeis e so' uma medicao separa:

  (a) o kernel INT8 nao esta' executando — "201 lineares trocados" e' saida de
      instrumento, nao prova de execucao;
  (b) o kernel executa, mas o GEMM nao e' onde o tempo esta'.

O `torch.profiler` responde as duas de uma vez, e mede tempo DE GPU por kernel,
que e' imune ao jitter de CPU que contaminou o cronometro de parede (o mesmo
braco deu 0,266 e 0,342 em rodadas diferentes).

    python scripts/perfil_passo_fish.py --braco base
    python scripts/perfil_passo_fish.py --braco int8_tudo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Familias de kernel, na ordem em que sao testadas. A primeira que casa leva.
# Sao substrings do nome do kernel CUDA; a lista veio de olhar a saida crua do
# profiler, nao de adivinhar nomes.
FAMILIAS = [
    ("GEMM int8",   ("int8", "i8gemm", "imma", "igemm", "cublasLt")),
    ("GEMM bf16",   ("gemm", "s16816", "bf16", "cutlass", "ampere_", "nn_align",
                     "nt_align", "tn_align")),
    ("elementwise", ("elementwise", "vectorized_elementwise", "unrolled_elementwise")),
    ("reducao",     ("reduce", "norm", "softmax", "mean", "sum")),
    ("copia",       ("copy", "cat", "index", "gather", "scatter", "pad", "slice",
                     "permute", "transpose", "contiguous")),
    ("otimizador",  ("adam", "foreach", "clip", "grad_norm")),
]


def familia(nome: str) -> str:
    baixo = nome.lower()
    for rotulo, chaves in FAMILIAS:
        if any(c.lower() in baixo for c in chaves):
            return rotulo
    return "outros"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--braco", default="base")
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--passos", type=int, default=6)
    ap.add_argument("--aquecer", type=int, default=12)
    ap.add_argument("--top", type=int, default=18)
    cli = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, str(ROOT / "vendor" / "fish-speech"))
    from bench_treino_fish import montar, um_passo

    modelo, opt, treinaveis = montar(cli, cli.braco)
    cfg = modelo.config
    nb, cb = cfg.num_codebooks, cfg.codebook_size

    inp = torch.randint(0, cb, (cli.batch, nb + 1, cli.max_length), device="cuda")
    inp[:, 0] = torch.randint(cfg.semantic_begin_id, cfg.semantic_end_id + 1,
                              (cli.batch, cli.max_length), device="cuda")
    lote = {"inp": inp,
            "mask": torch.zeros(cli.batch, cli.max_length, dtype=torch.bool,
                                device="cuda"),
            "labels": inp.clone()}

    for _ in range(cli.aquecer):
        um_passo(modelo, opt, lote, treinaveis, cfg)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(cli.passos):
            um_passo(modelo, opt, lote, treinaveis, cfg)
        torch.cuda.synchronize()

    # So' KERNEL DE VERDADE. `key_averages()` devolve tambem os wrappers de CPU
    # (`aten::mm`, `aten::linear`, `MmBackward0`), e o `device_time_total` deles
    # ja' inclui o tempo dos kernels que eles lancam. Somar os dois conta o mesmo
    # trabalho duas vezes: na primeira versao deste script o total deu 941 ms/passo
    # para um passo que o cronometro mede em 266 ms.
    from torch.profiler import DeviceType

    por_kernel: dict[str, tuple[float, int]] = {}
    for ev in prof.key_averages():
        if ev.device_type != DeviceType.CUDA:
            continue
        us = getattr(ev, "self_device_time_total", 0.0) or 0.0
        if us <= 0:
            continue
        anterior = por_kernel.get(ev.key, (0.0, 0))
        por_kernel[ev.key] = (anterior[0] + us, anterior[1] + ev.count)

    total = sum(v[0] for v in por_kernel.values())
    por_familia: dict[str, float] = defaultdict(float)
    for nome, (us, _) in por_kernel.items():
        por_familia[familia(nome)] += us

    p = cli.passos
    print(f"\n=== {cli.braco} — {cli.max_length} tokens, batch {cli.batch}, "
          f"{p} passos ===")
    print(f"tempo de GPU somado: {total / 1000 / p:.2f} ms/passo\n")

    print(f"{'familia':<14} {'ms/passo':>10} {'%':>7}")
    print("-" * 34)
    for rotulo, us in sorted(por_familia.items(), key=lambda x: -x[1]):
        print(f"{rotulo:<14} {us / 1000 / p:>10.2f} {us / total:>6.1%}")

    print(f"\n{'kernel':<58} {'ms/passo':>9} {'%':>6} {'chamadas':>9}")
    print("-" * 86)
    for nome, (us, n) in sorted(por_kernel.items(), key=lambda x: -x[1][0])[:cli.top]:
        curto = nome if len(nome) <= 57 else nome[:54] + "..."
        print(f"{curto:<58} {us / 1000 / p:>9.2f} {us / total:>5.1%} {n // p:>9}")

    saida = ROOT / "reports" / f"perfil_passo_{cli.braco}.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(
        {"braco": cli.braco, "tokens": cli.max_length, "batch": cli.batch,
         "ms_por_passo": total / 1000 / p,
         "familias": {k: v / 1000 / p for k, v in por_familia.items()},
         "kernels": {k: {"ms": v[0] / 1000 / p, "chamadas": v[1] // p}
                     for k, v in sorted(por_kernel.items(), key=lambda x: -x[1][0])}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
