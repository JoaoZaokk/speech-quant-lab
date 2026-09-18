"""Onde o INT8 acelera de verdade no bloco transformer do s2-pro.

POR QUE ESTE BENCH EXISTE
-------------------------
Em 2026-09-18 eu medi `torch._int_mm` e obtive 65 TOPS de 284 teoricos na 3090,
e concluí "int8 nao acelera aqui". A conclusao estava errada, e o erro tem nome:
eu medi UMA API, nao o FORMATO. `torch._int_mm` obriga a quantizar fora, chamar
o mm, e desquantizar fora — tres kernels, dois deles com trafego de memoria em
bf16. O GEMM int8 e' rapido; o sanduiche em volta dele nao.

O `comfy_kitchen::int8_linear` (cuBLASLt, `_C.int8_linear_m1` / `int8_linear`)
faz o oposto: quantiza a ativacao POR LINHA dentro do kernel e aplica a escala
no epilogo. Um lancamento, sem ida e volta a' memoria. O peso ja' chega em
int8 — que e' a condicao que o dono do projeto enunciou: "o verdadeiro int8 e'
ativado quando HA' o peso em INT8, senao e' so' gambiarra para calculo".

O QUE ESTE BENCH NAO RESPONDE
-----------------------------
Se serve para TREINAR. `comfy_kitchen::int8_linear` e' um `custom_op` sem
backward registrado — autograd nao atravessa. Aqui se mede FORWARD, que e' o
que decide se vale escrever o backward depois. Medir antes de construir.

FORMAS (s2-pro: dim 2560, 32 heads, 8 kv heads, head_dim 128, ffn 9728)
    wqkv   [6144, 2560]     (32+8+8)*128
    wo     [2560, 2560]
    w1/w3  [9728, 2560]     gate e up do SwiGLU
    w2     [2560, 9728]     down
    head   [155776, 2560]   a camada que custa 3,7x um bloco inteiro

    python scripts/bench_int8_kernel.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

# N, K de cada GEMM do s2-pro. `M` (linhas) e' o numero de tokens do passo.
FORMAS = [
    ("wqkv",  6144,   2560),
    ("wo",    2560,   2560),
    ("w1/w3", 9728,   2560),
    ("w2",    2560,   9728),
    ("head",  155776, 2560),
]


def cronometrar(fn, repete: int, aquecer: int = 5) -> float:
    """Mediana, nao media: um pico de coleta de lixo estraga a media."""
    for _ in range(aquecer):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(repete):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    return st.median(t)


def erro_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """L2 relativo. Velocidade sem precisao nao e' ganho, e' troca."""
    return float((a.float() - b.float()).norm() / b.float().norm())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", default="145,580,2048",
                    help="M de cada teste; 145 e' a mediana real do pack, "
                         "580 e' batch 4, 2048 e' para ver a curva")
    ap.add_argument("--repete", type=int, default=30)
    cli = ap.parse_args()

    from comfy_kitchen.backends import cuda as ckc
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as INT8

    print(f"{torch.cuda.get_device_name(0)}  sm{''.join(map(str, torch.cuda.get_device_capability(0)))}"
          f"  torch {torch.__version__}  cublasLt={ckc._CUBLASLT_AVAILABLE}")

    linhas = []
    for M in [int(x) for x in cli.tokens.split(",")]:
        print(f"\n{'='*86}\nM = {M} tokens\n{'='*86}")
        print(f"{'GEMM':<8} {'N':>7} {'K':>6} {'bf16':>9} {'int8 canal':>11} "
              f"{'int8 tensor':>12} {'_int_mm':>9} {'ganho':>7} {'erroL2':>9}")
        print("-" * 86)

        for nome, N, K in FORMAS:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02

            ref = F.linear(x, w)
            t_bf16 = cronometrar(lambda: F.linear(x, w), cli.repete)

            # ---- int8 per-channel (escala por linha do peso) ----
            qw_c, p_c = INT8.quantize(w, is_weight=True, per_channel=True)
            s_c = p_c.scale.to(torch.float32).reshape(-1).contiguous()
            try:
                out_c = ckc.int8_linear(x, qw_c, s_c, out_dtype=torch.bfloat16)
                t_c, e_c = cronometrar(
                    lambda: ckc.int8_linear(x, qw_c, s_c, out_dtype=torch.bfloat16),
                    cli.repete), erro_rel(out_c, ref)
            except Exception as exc:
                t_c, e_c = None, None
                print(f"  ! canal falhou em {nome}: {type(exc).__name__}: {exc}")

            # ---- int8 tensorwise (uma escala para o peso inteiro) ----
            qw_t, p_t = INT8.quantize(w, is_weight=True, per_channel=False)
            s_t = p_t.scale.to(torch.float32).reshape(-1).contiguous()
            try:
                out_t = ckc.int8_linear(x, qw_t, s_t, out_dtype=torch.bfloat16)
                t_t, e_t = cronometrar(
                    lambda: ckc.int8_linear(x, qw_t, s_t, out_dtype=torch.bfloat16),
                    cli.repete), erro_rel(out_t, ref)
            except Exception as exc:
                t_t, e_t = None, None
                print(f"  ! tensor falhou em {nome}: {type(exc).__name__}: {exc}")

            # ---- torch._int_mm: o que eu medi antes, para contraste ----
            # O sanduiche COMPLETO, que e' o que um usuario de verdade paga:
            # quantizar a ativacao, o mm, e reescalar de volta para bf16.
            xi = (x / x.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) * 127)
            xi = xi.round().to(torch.int8)
            wi = qw_c.t().contiguous()

            def _sanduiche():
                s = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 127
                q = (x / s).round().to(torch.int8)
                acc = torch._int_mm(q, wi)
                return (acc.float() * s.float() * s_c.float()).to(torch.bfloat16)

            try:
                t_i = cronometrar(_sanduiche, cli.repete)
            except Exception:
                t_i = None

            melhor = min([t for t in (t_c, t_t) if t is not None], default=None)
            ganho = t_bf16 / melhor if melhor else 0.0

            def f(v, casas=3):
                return f"{v:.{casas}f}" if v is not None else "—"

            print(f"{nome:<8} {N:>7} {K:>6} {f(t_bf16):>9} {f(t_c):>11} "
                  f"{f(t_t):>12} {f(t_i):>9} {ganho:>6.2f}x {f(e_c, 5):>9}")

            linhas.append({"M": M, "gemm": nome, "N": N, "K": K,
                           "ms_bf16": t_bf16, "ms_int8_canal": t_c,
                           "ms_int8_tensor": t_t, "ms_int_mm_sanduiche": t_i,
                           "ganho": ganho, "erro_canal": e_c, "erro_tensor": e_t})
            del x, w, qw_c, qw_t
            torch.cuda.empty_cache()

    saida = ROOT / "reports" / "bench_int8_kernel.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(linhas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
