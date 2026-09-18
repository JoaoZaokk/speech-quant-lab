"""Mede s/step do TREINO do Fish, um acelerador de cada vez.

POR QUE NAO SERVE O `bench_fish_seqlen.py`
------------------------------------------
Aquele mede INFERENCIA, que e' outro perfil: batch 1, decode autorregressivo,
KV cache, sem backward. No treino a sequencia vem inteira de uma vez, ha'
backward, e o gradient checkpointing REFAZ o forward de cada bloco. Conclusao
de um nao transfere para o outro — ja' aconteceu neste projeto com o
`torch.compile`, que da' 2,9x no decode do Fish e PIORA o LFM em 8%.

O QUE ESTA E O QUE NAO ESTA LIGADO HOJE (levantado em 2026-09-18)
-----------------------------------------------------------------
    triton 3.7.1        instalado, so' entra via torch.compile
    torch.compile       NAO usado no treino — `train_fish.py` nem tem a flag
    sageattention       instalado, ZERO import no vendor
    flash_attn          instalado, ZERO import no vendor
    torchao             instalado, ZERO import
    SDPA flash nativo   so' no prefill e so' quando `mask is None`
    RMSNorm             Python, com `.float()` por chamada, 36 camadas x 2
    SwiGLU              tres GEMMs soltas: w2(silu(w1(x)) * w3(x))
    gradient ckpt       LIGADO — troca memoria por ~30% de tempo

O treino de 3h02 de 18/09 rodou com NENHUM deles.

COMO MEDIR SEM MENTIR PARA SI MESMO
-----------------------------------
- `--aquecer` passos descartados antes de cronometrar. Com `torch.compile` os
  primeiros passos incluem a compilacao, que e' custo unico e nao deve entrar
  na media de um treino de milhares de steps.
- mediana, nao media: um pico de swap ou de coleta de lixo estraga a media.
- `torch.cuda.max_memory_allocated`, nao `nvidia-smi`: o segundo inclui o cache
  do alocador, que nao devolve memoria e nao mede uso vivo.
- MESMO dado e MESMA seed em todos os bracos.
- `--repetir` ALTERNANDO os bracos, e olhar o espalhamento antes do ganho.

O MINIMO, NAO A MEDIANA, QUANDO HA' INTERFERENCIA
-------------------------------------------------
MEDIDO em 2026-09-18: o `MsMpEng.exe` (Windows Defender) sobe a 100% de um
nucleo varrendo os 10 GB de safetensors que cada montagem le'. Este passo esta'
limitado por CPU — 217 ms de GPU num passo de 266 ms de parede — entao a
varredura entra direto no cronometro.

A interferencia e' ADITIVA e INTERMITENTE: ela so' pode ATRASAR um passo, nunca
adiantar. Sob esse regime o MINIMO e' o estimador do tempo verdadeiro e a
mediana e' um chute sobre quanto o Defender trabalhou. Foi o minimo que se
manteve firme entre execucoes (`compile` deu 0,184 nas duas baterias) enquanto a
mediana da mesma medicao variou de 0,184 para 0,247.

A tabela final imprime mediana, minimo e maximo lado a lado de proposito:
comparar os MINIMOS, e usar o espalhamento so' para saber se a bateria toda
presta.

    python scripts/bench_treino_fish.py --lista base,compile
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Diretorio do checkpoint do S2 Pro (o que tem `config.json`, os shards e o
# tokenizer). Aponte com `FISH_CKPT` ou com `--ckpt`.
CKPT = os.environ.get("FISH_CKPT", "checkpoints/s2-pro")

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def montar(cli, braco: str):
    """Monta modelo + otimizador com o acelerador do braco ligado."""
    import torch

    from fish_speech.models.text2semantic.llama import BaseTransformer
    from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora

    modelo = BaseTransformer.from_pretrained(
        getattr(cli, "ckpt", None) or CKPT,
        load_weights=True, max_length=cli.max_length, lora_config=None)

    # `gradckpt_off` mede quanto o gradient checkpointing custa. Ele troca VRAM
    # por tempo refazendo o forward de cada bloco no backward; se a VRAM sobrar,
    # desligar e' ganho puro.
    if "gradckpt" in braco and "off" in braco:
        modelo.config.use_gradient_checkpointing = False

    # `ckptN` = checkpoint SELETIVO, uma camada a cada N. MEDIDO: o recompute e'
    # ~26 ms dos ~123 ms de GPU do passo com INT8, e nao produz nada — existe so'
    # para caber na VRAM. Desligar tudo da' OOM; desligar METADE talvez caiba.
    import re as _re
    if (m := _re.search(r"ckpt(\d+)", braco)):
        modelo.config.gradient_checkpointing_cada = int(m.group(1))

    # `memN` = ORCAMENTO DE MEMORIA DE ATIVACAO do AOTAutograd, em porcento.
    #
    # E' a versao automatica do "selective activation checkpointing": em vez de
    # eu listar quais operacoes salvar, o particionador estima FLOPs por byte de
    # cada tensor salvo e resolve uma mochila — guarda o que e' caro de refazer
    # e barato de guardar, recomputa o resto.
    #
    # So' faz sentido com `gradckpt_off`: o `checkpoint()` manual do vendor joga
    # fora o bloco INTEIRO antes do particionador ver qualquer coisa, entao com
    # ele ligado nao sobra decisao para o orcamento tomar.
    #
    # 1,0 = salva tudo que couber (menos recompute, mais VRAM).
    if (m := _re.search(r"mem(\d+)", braco)):
        import torch._functorch.config as _fc
        _fc.activation_memory_budget = int(m.group(1)) / 100

    setup_lora(modelo, LoraConfig(r=32, lora_alpha=16, lora_dropout=0.1,
                                  target_modules=["slow_attention", "slow_mlp"]))
    import loralib
    loralib.mark_only_lora_as_trainable(modelo, bias="none")

    modelo = modelo.cuda().to(torch.bfloat16)
    modelo.train()

    # INT8 vem DEPOIS do `.cuda().to(bfloat16)` — a quantizacao le' o peso ja'
    # no device e no dtype finais — e ANTES do `torch.compile`, porque o grafo
    # precisa ser tracado sobre os modulos que vao mesmo rodar.
    if braco.startswith("int8"):
        sys.path.insert(0, str(ROOT / "src"))
        from accel.int8_linear import aplicar

        # Cada subconjunto responde uma pergunta diferente: onde o INT8 paga o
        # ganho sem cobrar precisao. Nomes somam — `int8_mlp_cabeca_compile`.
        partes = set(braco.split("_"))
        # Sem qualificador, `int8` sozinho quer dizer tudo — senao um `--lista
        # int8_compile` desligaria o proprio acelerador em silencio e mediria
        # o compile puro achando que mede INT8.
        if not partes & {"mlp", "attn", "cabeca", "tudo"}:
            partes.add("tudo")
        alvos: tuple[str, ...] = ()
        if partes & {"mlp", "tudo"}:
            alvos += ("w1", "w3", "w2")
        if partes & {"attn", "tudo"}:
            alvos += ("wqkv", "wo")
        aplicar(modelo, alvos=alvos,
                cabeca=bool(partes & {"cabeca", "tudo"}),
                backward_int8="bwd" in partes)

    if "compile" in braco:
        if "gradckpt_off" in braco:
            modelo.config.use_gradient_checkpointing = False
        # MEDIDO em 2026-09-18: 88,7% dos 11.290 kernels de um passo duram menos
        # de 10 us, e um `cudaLaunchKernel` custa 5-10 us. Paga-se mais para
        # PEDIR do que para receber. Dois remedios, e sao diferentes:
        #
        #   `default`         funde elementwise em kernels Triton — menos
        #                     kernels. Corta 11.290 -> 3.982.
        #   `reduce-overhead` liga CUDA Graphs: grava a sequencia de lancamentos
        #                     uma vez e reproduz com UM comando ao driver. Ataca
        #                     o custo POR LANCAMENTO, nao a quantidade.
        #
        # `max-autotune` custa minutos de compilacao a mais e so' paga em treino
        # longo; nao entra aqui.
        modo = "reduce-overhead" if "graphs" in braco else None
        modelo = torch.compile(modelo, mode=modo)

    treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(treinaveis, lr=2e-5, weight_decay=0, betas=(0.9, 0.95))
    return modelo, opt, treinaveis


def um_passo(modelo, opt, lote, treinaveis, cfg):
    """Espelha o `_step` do `lit_module.py`, incluindo a loss dos codebooks.

    O `labels` NAO e' opcional: o `DualARTransformer.forward` usa `labels[:, 0]`
    para descobrir quais posicoes sao semanticas e so' essas entram no Fast AR.
    Passar `None` nao "mede o forward sem a loss" — quebra. E medir so' a
    `base_loss` mediria metade do grafo, deixando de fora as 4 camadas do Fast
    e o backward delas.
    """
    import torch
    import torch.nn.functional as F

    saida = modelo(inp=lote["inp"], key_padding_mask=lote["mask"],
                   labels=lote["labels"])

    tl = saida.token_logits
    base = F.cross_entropy(tl.view(-1, tl.size(-1)),
                           lote["labels"][:, 0].reshape(-1), ignore_index=-100)

    cl = saida.codebook_logits
    token_ids = lote["labels"][:, 0]
    sem = (token_ids >= cfg.semantic_begin_id) & (token_ids <= cfg.semantic_end_id)
    alvos_cb = lote["labels"][:, 1:1 + cfg.num_codebooks].permute(0, 2, 1)[sem]
    semantica = F.cross_entropy(cl.reshape(-1, cl.size(-1)),
                                alvos_cb.reshape(-1), ignore_index=-100)

    perda = base + semantica
    perda.backward()
    torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(perda.detach())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lista", default="base,compile,gradckpt_off",
                    help="bracos separados por virgula")
    ap.add_argument("--passos", type=int, default=12, help="passos cronometrados")
    ap.add_argument("--aquecer", type=int, default=20,
                    help="passos descartados antes de cronometrar. Com compile "
                         "os primeiros incluem a compilacao, que e' custo unico. "
                         "MEDIDO em 2026-09-18: com 5 passos o MESMO braco deu "
                         "0,263 / 0,263 / 0,313 s em tres processos — 19% de "
                         "espalhamento, maior que qualquer ganho que se queira "
                         "medir. A 3090 fica em 210 MHz ociosa e a rampa ate' os "
                         "2100 MHz nao cabe em 5 passos.")
    ap.add_argument("--repetir", type=int, default=1,
                    help="repete a lista inteira N vezes, ALTERNANDO os bracos "
                         "(A,B,A,B) em vez de agrupar (A,A,B,B). A alternancia e' "
                         "que cancela deriva termica e de clock: agrupado, uma "
                         "deriva lenta vira 'ganho' do braco que rodou por ultimo.")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1,
                    help="MEDIDO em 2026-09-18: com 145 tokens e batch 1 a 3090 "
                         "fica subocupada — 36 camadas de 2560 dim processando "
                         "145 posicoes. O overhead de lancamento de kernel "
                         "domina, e e' por isso que o compile rende 1,77x aqui "
                         "contra 1,34x em 512 tokens. Subir o batch ataca a "
                         "ociosidade, nao o overhead.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ckpt", default=CKPT,
                    help="diretorio do S2 Pro; tambem le' de $FISH_CKPT")
    cli = ap.parse_args()

    import torch

    sys.path.insert(0, str(ROOT / "vendor" / "fish-speech"))

    bracos = [b.strip() for b in cli.lista.split(",") if b.strip()]
    ordem = [(r, b) for r in range(cli.repetir) for b in bracos]

    resultados = []
    for rodada, braco in ordem:
        torch.manual_seed(cli.seed)
        # `gc.collect()` antes do `empty_cache()`: sem ele o grafo de autograd do
        # braco anterior sobrevive por referencia circular e o `empty_cache()` nao
        # tem o que devolver. MEDIDO: o segundo braco dava OOM com 22,21 GB ainda
        # alocados apos `del modelo, opt`.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n=== {braco}"
              + (f" (rodada {rodada + 1}/{cli.repetir})" if cli.repetir > 1 else "")
              + " ===", flush=True)

        t0 = time.perf_counter()
        modelo, opt, treinaveis = montar(cli, braco)
        carga = time.perf_counter() - t0
        n_tr = sum(p.numel() for p in treinaveis)
        print(f"carga {carga:.1f}s  treinaveis {n_tr/1e6:.2f}M", flush=True)

        # Lote sintetico: a pergunta e' o CUSTO POR PASSO, e isso nao depende do
        # conteudo — depende de forma, dtype e caminho de kernel. Dado real aqui
        # so' acrescentaria I/O de disco a' medicao.
        cfg = modelo.config
        V, nb, cb = cfg.vocab_size, cfg.num_codebooks, cfg.codebook_size
        # As duas linhas do `inp` tem FAIXAS DIFERENTES, e misturar da'
        # `device-side assert` no embedding:
        #   linha 0      token de texto/semantica, ate' vocab_size
        #   linhas 1..n  codebooks, ate' codebook_size (4096) — o `embed` soma
        #                `i * codebook_size` a cada uma, entao valor acima de
        #                4096 invade a faixa do codebook seguinte e estoura.
        # Uso a faixa SEMANTICA na linha 0 para o caminho ser o mesmo do treino
        # real (`vq_masks` liga quando o token e' semantico).
        inp = torch.randint(0, cb, (cli.batch, nb + 1, cli.max_length), device="cuda")
        inp[:, 0] = torch.randint(cfg.semantic_begin_id, cfg.semantic_end_id + 1,
                                  (cli.batch, cli.max_length), device="cuda")
        lote = {
            "inp": inp,
            "mask": torch.zeros(cli.batch, cli.max_length, dtype=torch.bool, device="cuda"),
            # `labels` tem a mesma forma do `inp` — o forward le' `labels[:, 0]`
            # para achar as posicoes semanticas e `labels[:, 1:]` como alvo dos
            # codebooks.
            "labels": inp.clone(),
        }

        for _ in range(cli.aquecer):
            um_passo(modelo, opt, lote, treinaveis, cfg)
        torch.cuda.synchronize()

        tempos = []
        for i in range(cli.passos):
            t = time.perf_counter()
            um_passo(modelo, opt, lote, treinaveis, cfg)
            torch.cuda.synchronize()
            tempos.append(time.perf_counter() - t)

        # O MINIMO dos passos, nao a mediana. A interferencia externa (varredura
        # de antivirus, outro processo, rampa de clock) so' pode ATRASAR um
        # passo — nenhuma delas faz um passo terminar antes do que o hardware
        # permite. Sob ruido puramente aditivo o minimo estima o tempo real e a
        # mediana estima "tempo real + quanto a maquina estava ocupada".
        rapido, med = min(tempos), st.median(tempos)
        vram = torch.cuda.max_memory_allocated() / 2**30
        print(f"  {rapido:.3f} s/passo (minimo de {cli.passos}, mediana "
              f"{med:.3f})   VRAM {vram:.2f} GB")
        resultados.append({"braco": braco, "rodada": rodada,
                           "s_por_passo": rapido, "mediana_s": med,
                           "vram_gb": vram, "carga_s": carga, "tempos": tempos})

        del modelo, opt, treinaveis, lote
        torch.cuda.empty_cache()

    # Agrega por braco: a mediana das rodadas, e o ESPALHAMENTO entre elas.
    # O espalhamento nao e' enfeite — e' a regua. Um ganho menor que o espaco
    # entre as rodadas do proprio braco de referencia nao foi medido, foi
    # sorteado.
    por_braco: dict[str, list[float]] = {}
    for r in resultados:
        por_braco.setdefault(r["braco"], []).append(r["s_por_passo"])

    # O ganho sai do MINIMO: a interferencia externa so' atrasa, nunca adianta.
    ref_min = min(por_braco[bracos[0]])
    print("\n" + "=" * 82)
    print(f"{'braco':24s} {'min':>8s} {'ganho':>7s} {'mediana':>9s} {'max':>8s} "
          f"{'espalha':>8s} {'VRAM':>9s}")
    for nome, vals in por_braco.items():
        vram = max(r["vram_gb"] for r in resultados if r["braco"] == nome)
        espalha = (max(vals) - min(vals)) / min(vals) if len(vals) > 1 else 0.0
        print(f"{nome:24s} {min(vals):8.3f} {ref_min / min(vals):6.2f}x "
              f"{st.median(vals):9.3f} {max(vals):8.3f} {espalha:7.1%} "
              f"{vram:8.2f} GB")
    print("=" * 82)
    print("ganho medido sobre o MINIMO de cada braco — ver o cabecalho do arquivo")
    if cli.repetir > 1:
        pior = max((max(v) - min(v)) / st.median(v) for v in por_braco.values())
        print(f"maior espalhamento dentro de um braco: {pior:.1%} — "
              f"ganho abaixo disso nao e' conclusivo")

    saida = ROOT / "reports" / "bench_treino_fish.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"-> {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
