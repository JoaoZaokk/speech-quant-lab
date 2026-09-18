"""Quanto o INT8 muda o que o Fish calcula — e, sobretudo, o que ele APRENDE.

POR QUE ISTO VEM ANTES DE QUALQUER TREINO
-----------------------------------------
`scripts/bench_treino_fish.py` mede se o passo fica mais rapido. Nao mede se o
passo continua CERTO. Velocidade que estraga o treino nao e' ganho, e o projeto
ja' pagou por metrica que mediu a coisa errada mais de uma vez.

O erro de um GEMM isolado e' 1,3% (`scripts/bench_int8_kernel.py`). Isso NAO
responde a pergunta: sao 36 camadas, o erro atravessa todas, e o que decide o
treino nao e' o logit — e' o GRADIENTE que chega na LoRA.

O QUE SE MEDE, EM ORDEM DE IMPORTANCIA
--------------------------------------
1. `grad` da LoRA: o erro relativo, parametro a parametro, entre o gradiente
   que o treino bf16 produziria e o que o INT8 produz. E' a unica quantidade
   que muda os pesos.
2. cosseno entre os dois gradientes: erro de MAGNITUDE o otimizador absorve
   (Adam normaliza pelo segundo momento); erro de DIRECAO, nao. Um cosseno de
   0,999 com norma 10% menor e' inofensivo; 0,9 de cosseno nao e'.
3. logits e loss: o efeito visivel, para contexto.

    python scripts/erro_int8_fish.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--braco", default="int8_tudo",
                    help="braco int8 a comparar contra o 'base'")
    cli = ap.parse_args()

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(ROOT / "vendor" / "fish-speech"))
    from bench_treino_fish import montar

    def rodar(braco: str):
        """Um passo completo, devolvendo logits, loss e o grad de cada LoRA.

        A seed e' refixada antes de montar E antes do passo: o `lora_dropout`
        de 0,1 sorteia mascara, e mascaras diferentes produziriam gradientes
        diferentes por motivo que nao tem nada a ver com INT8.
        """
        torch.manual_seed(cli.seed)
        modelo, opt, treinaveis = montar(cli, braco)
        cfg = modelo.config
        nb, cb = cfg.num_codebooks, cfg.codebook_size

        torch.manual_seed(cli.seed)
        inp = torch.randint(0, cb, (cli.batch, nb + 1, cli.max_length), device="cuda")
        inp[:, 0] = torch.randint(cfg.semantic_begin_id, cfg.semantic_end_id + 1,
                                  (cli.batch, cli.max_length), device="cuda")
        lote = {"inp": inp,
                "mask": torch.zeros(cli.batch, cli.max_length, dtype=torch.bool,
                                    device="cuda"),
                "labels": inp.clone()}

        torch.manual_seed(cli.seed)
        saida = modelo(inp=lote["inp"], key_padding_mask=lote["mask"],
                       labels=lote["labels"])
        tl = saida.token_logits
        base = F.cross_entropy(tl.view(-1, tl.size(-1)),
                               lote["labels"][:, 0].reshape(-1), ignore_index=-100)
        cl = saida.codebook_logits
        tok = lote["labels"][:, 0]
        sem = (tok >= cfg.semantic_begin_id) & (tok <= cfg.semantic_end_id)
        alvos = lote["labels"][:, 1:1 + cfg.num_codebooks].permute(0, 2, 1)[sem]
        semantica = F.cross_entropy(cl.reshape(-1, cl.size(-1)),
                                    alvos.reshape(-1), ignore_index=-100)
        perda = base + semantica
        perda.backward()

        grads = {n: p.grad.detach().float().clone()
                 for n, p in modelo.named_parameters()
                 if p.requires_grad and p.grad is not None}
        out = (tl.detach().float().clone(), float(perda), grads)

        del modelo, opt, treinaveis, lote, saida, tl, cl
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return out

    print("=== base (bf16) ===", flush=True)
    logits_ref, perda_ref, grad_ref = rodar("base")
    print(f"=== {cli.braco} ===", flush=True)
    logits_q, perda_q, grad_q = rodar(cli.braco)

    rel_logits = float((logits_q - logits_ref).norm() / logits_ref.norm())

    linhas = []
    for nome in sorted(grad_ref):
        a, b = grad_ref[nome], grad_q.get(nome)
        if b is None:
            continue
        na, nb_ = a.norm(), b.norm()
        linhas.append({
            "param": nome,
            "rel": float((b - a).norm() / na) if na > 0 else 0.0,
            "cos": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
            "razao_norma": float(nb_ / na) if na > 0 else 0.0,
        })

    rels = sorted(x["rel"] for x in linhas)
    coss = sorted(x["cos"] for x in linhas)
    n = len(linhas)

    print(f"\n{'='*70}")
    print(f"logits:  erro relativo L2 = {rel_logits:.4f}")
    print(f"loss:    bf16 {perda_ref:.4f}   int8 {perda_q:.4f}   "
          f"delta {perda_q - perda_ref:+.4f}")
    print(f"\ngradiente da LoRA ({n} tensores):")
    print(f"  erro relativo   mediana {rels[n//2]:.4f}   "
          f"p90 {rels[int(n*0.9)]:.4f}   pior {rels[-1]:.4f}")
    print(f"  cosseno         mediana {coss[n//2]:.5f}   "
          f"p10 {coss[int(n*0.1)]:.5f}   pior {coss[0]:.5f}")
    print("\n  piores 5 por cosseno (direcao e' o que o Adam nao corrige):")
    for x in sorted(linhas, key=lambda x: x["cos"])[:5]:
        print(f"    {x['param'][-52:]:<52} cos {x['cos']:.5f}  "
              f"rel {x['rel']:.4f}  norma x{x['razao_norma']:.3f}")
    print("=" * 70)

    saida = ROOT / "reports" / "erro_int8_fish.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(
        {"braco": cli.braco, "tokens": cli.max_length,
         "erro_logits": rel_logits, "loss_bf16": perda_ref, "loss_int8": perda_q,
         "grad": linhas}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
