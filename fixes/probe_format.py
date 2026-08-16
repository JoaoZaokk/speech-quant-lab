"""Mede a perda do s2-pro CRU em cada formato de prompt possivel.

POR QUE ISTO EXISTE
-------------------
O treino do Fish nao estava aprendendo. Com LoRA so' no Fast AR, 100 passos
deixaram `val/loss` em 28,2 — parado. E os componentes da perda contam o
porque':

    base_loss    11,95      ln(155776) = 11,958   <- ACASO PURO
    semantic_loss 16,85     ln(4096)   =  8,318   <- DUAS VEZES o acaso

Modelo pre-treinado nao fica no acaso na propria tarefa. Isso nao e' "falta
finetune", e' entrada fora da distribuicao.

E de fato o repo monta a sequencia de DUAS formas diferentes:

  TREINO (`AutoTextSemanticInstructionIterableDataset.pack_sentences`):
      Speak out the provided text.
      <|speaker:user|> {texto}<|im_end|>
      <|speaker:assistant|> <|voice|>{codigos VQ}<|im_end|>

  INFERENCIA (`text2semantic/inference.py`, com referencia):
      <|im_start|>system
      convert the provided text to speech reference to the following:

      Text:
      <|speaker:0|>{texto da referencia}

      Speech:
      {codigos VQ da referencia}<|im_end|>
      <|im_start|>user
      {texto alvo}<|im_end|>
      <|im_start|>assistant
      <|voice|>{gera}

Nao e' a mesma coisa: papeis com `<|im_start|>` contra strings literais
`<|speaker:user|>`, texto de sistema diferente, e — o principal — a inferencia
SEMPRE tem audio de referencia e o treino NUNCA tem.

Este script mede as tres variantes no mesmo modelo, sem treinar nada:

    atual   o que o dataset do repo produz hoje
    chat    papeis system/user/assistant, sem referencia
    ref     formato da inferencia, com clipe de referencia do mesmo locutor

Se `ref` derrubar `base_loss` de ~12 para a casa de 1-3, a causa esta' provada
e a conclusao e' direta: treinar no formato `atual` ensina o modelo um prompt
que ele nunca vai receber.

    python fixes/probe_format.py --ckpt /caminho/para/s2-pro --protos /caminho/para/protos
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# O `fish_speech` precisa estar importavel. Se voce tem um checkout do
# fish-speech em vez de instalacao, aponte com --fish-speech ou com a variavel
# FISH_SPEECH_PATH; sem isso, assume que ja' esta' no PYTHONPATH.
_fs = os.environ.get("FISH_SPEECH_PATH")
if _fs:
    sys.path.insert(0, str(Path(_fs).resolve()))

# caminho do checkpoint do s2-pro baixado de https://huggingface.co/fishaudio/s2-pro
CKPT = None
CODEBOOK_PAD_TOKEN_ID = 0


def montar(encoded, num_codebooks: int):
    """(tokens, labels) no layout (num_codebooks+1, L) que o modelo espera.

    Copiado de `pack_sentences` do repo — as duas implementacoes do dataset
    fazem exatamente isto, entao replicar aqui mantem a medida comparavel.
    """
    import torch

    tokens_raw = encoded.tokens
    tokens = torch.zeros((num_codebooks + 1, len(tokens_raw)), dtype=torch.int)
    tokens[0] = tokens_raw

    vq = [p for p in encoded.vq_parts]
    if vq:
        vq = torch.cat(vq, dim=1)
        tokens[1:, encoded.vq_mask_tokens] = vq

    labels_raw = encoded.labels
    labels = torch.full((num_codebooks + 1, len(labels_raw)), -100, dtype=torch.int)
    labels[0, :] = labels_raw
    if len(vq):
        labels[1:, encoded.vq_mask_labels] = vq
    labels[1:, -1:] = CODEBOOK_PAD_TOKEN_ID

    return tokens.long(), labels.long()


def seq_atual(tok, alvo, ref, num_codebooks: int):
    """Formato que o `AutoTextSemanticInstructionIterableDataset` produz."""
    import torch

    from fish_speech.content_sequence import ContentSequence, TextPart, VQPart
    from fish_speech.text.clean import clean_text

    seq = ContentSequence()
    seq.append(TextPart(text="Speak out the provided text."))
    seq.append(TextPart(text=f"<|speaker:user|> {clean_text(alvo['texto'])}"), add_end=True)
    codes = torch.tensor(alvo["codes"]).to(torch.int32)
    seq.append([TextPart(text="<|speaker:assistant|> <|voice|>"),
                VQPart(codes=codes, cal_loss=True)], add_end=True)
    return montar(seq.encode(tokenizer=tok), num_codebooks)


def seq_chat(tok, alvo, ref, num_codebooks: int):
    """Papeis de conversa, sem referencia. O que o dataset NAO-iteravel faz."""
    import torch

    from fish_speech.content_sequence import TextPart, VQPart
    from fish_speech.conversation import Conversation, Message
    from fish_speech.text.clean import clean_text

    conv = Conversation([
        Message(role="system",
                parts=[TextPart(text="convert the provided text to speech")],
                cal_loss=False),
        Message(role="user",
                parts=[TextPart(text=clean_text(alvo["texto"]))],
                cal_loss=False),
        Message(role="assistant",
                parts=[VQPart(codes=torch.tensor(alvo["codes"]).to(torch.int32),
                              cal_loss=True)],
                cal_loss=True, modality="voice"),
    ])
    # `Conversation.encode` repassa `max_length=` para `ContentSequence.encode`,
    # que nao aceita esse parametro nesta versao do vendor — TypeError garantido.
    # `to_content_sequence()` monta as mesmas partes e desvia do bug.
    return montar(conv.to_content_sequence().encode(tokenizer=tok), num_codebooks)


def seq_ref(tok, alvo, ref, num_codebooks: int):
    """Formato da inferencia: referencia do mesmo locutor no system prompt."""
    import torch

    from fish_speech.content_sequence import TextPart, VQPart
    from fish_speech.conversation import Conversation, Message
    from fish_speech.text.clean import clean_text

    if ref is None:
        return None

    conv = Conversation([
        Message(
            role="system",
            parts=[
                TextPart(text="convert the provided text to speech reference to "
                              "the following:\n\nText:\n"),
                TextPart(text=f"<|speaker:0|>{clean_text(ref['texto'])}"),
                TextPart(text="\n\nSpeech:\n"),
                VQPart(codes=torch.tensor(ref["codes"]).to(torch.int32), cal_loss=False),
            ],
            cal_loss=False),
        Message(role="user",
                parts=[TextPart(text=clean_text(alvo["texto"]))],
                cal_loss=False),
        Message(role="assistant",
                parts=[VQPart(codes=torch.tensor(alvo["codes"]).to(torch.int32),
                              cal_loss=True)],
                cal_loss=True, modality="voice"),
    ])
    # `Conversation.encode` repassa `max_length=` para `ContentSequence.encode`,
    # que nao aceita esse parametro nesta versao do vendor — TypeError garantido.
    # `to_content_sequence()` monta as mesmas partes e desvia do bug.
    return montar(conv.to_content_sequence().encode(tokenizer=tok), num_codebooks)


FORMATOS = {"atual": seq_atual, "chat": seq_chat, "ref": seq_ref}


def carregar_grupos(protos: Path) -> list[list[dict]]:
    from fish_speech.datasets.protos.text_data_stream import read_pb_stream

    arquivos = sorted(protos.rglob("*.protos")) + sorted(protos.rglob("*.proto"))
    if not arquivos:
        raise SystemExit(f"sem protos em {protos}")

    grupos = []
    for a in arquivos:
        with open(a, "rb") as f:
            for g in read_pb_stream(f):
                frases = []
                for s in g.sentences:
                    if not s.texts or not s.semantics:
                        continue
                    frases.append({
                        "texto": s.texts[0],
                        "codes": [list(x.values) for x in s.semantics],
                    })
                if frases:
                    grupos.append(frases)
    return grupos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protos", required=True,
                    help="diretorio de protos exportado pelo tools/build_dataset.py do fish-speech")
    ap.add_argument("--ckpt", default=CKPT, required=CKPT is None,
                    help="diretorio do checkpoint fishaudio/s2-pro")
    ap.add_argument("--fish-speech", default=None,
                    help="checkout do fish-speech, se nao estiver instalado")
    ap.add_argument("--lora-ckpt", default=None,
                    help="checkpoint .ckpt do Lightning para medir um modelo ja' treinado")
    ap.add_argument("--n", type=int, default=32, help="amostras por formato")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # tem que entrar no path ANTES do primeiro import de `fish_speech`, que
    # acontece logo abaixo — os imports deste script sao todos tardios de
    # proposito, justamente para isso funcionar
    if args.fish_speech:
        sys.path.insert(0, str(Path(args.fish_speech).resolve()))

    import torch
    import torch.nn.functional as F

    from fish_speech.models.text2semantic.llama import BaseTransformer

    modelo = BaseTransformer.from_pretrained(
        path=args.ckpt, load_weights=True, max_length=args.max_length, lora_config=None,
    )
    modelo = modelo.to(device="cuda", dtype=torch.bfloat16).eval()
    tok = modelo.tokenizer
    nc = modelo.config.num_codebooks
    print(f"modelo carregado: {nc} codebooks, max_seq_len {modelo.config.max_seq_len}")

    grupos = carregar_grupos(Path(args.protos))
    com_par = [g for g in grupos if len(g) >= 2]
    print(f"{len(grupos)} grupos, {len(com_par)} com >=2 clipes "
          f"(so' estes servem para o formato `ref`)")

    # MESMOS pares para os tres formatos. Sortear por formato compararia
    # formato com sorte, nao formato com formato.
    rng = random.Random(args.seed)
    pares = []
    for _ in range(args.n):
        g = rng.choice(com_par)
        i, j = rng.sample(range(len(g)), 2)
        pares.append((g[j], g[i]))   # (alvo, referencia)

    resultados: dict[str, dict] = {}
    for nome, fn in FORMATOS.items():
        somas = {"base": 0.0, "sem": 0.0, "top5": 0.0}
        n_ok = 0
        for alvo, ref in pares:
            feito = fn(tok, alvo, ref, nc)
            if feito is None:
                continue
            tokens, labels = feito
            tokens = tokens[:, :args.max_length].unsqueeze(0).cuda()
            labels = labels[:, :args.max_length].unsqueeze(0).cuda()
            mask = torch.zeros((1, tokens.size(2)), dtype=torch.bool, device="cuda")

            with torch.no_grad():
                out = modelo(inp=tokens, key_padding_mask=mask, labels=labels)

            base = F.cross_entropy(
                out.token_logits.view(-1, out.token_logits.size(-1)),
                labels[:, 0].reshape(-1), ignore_index=-100)

            ids = labels[:, 0]
            sem_mask = ((ids >= tok.semantic_begin_id) & (ids <= tok.semantic_end_id))
            cb = labels[:, 1:1 + nc].permute(0, 2, 1)[sem_mask]
            sem = F.cross_entropy(
                out.codebook_logits.reshape(-1, out.codebook_logits.size(-1)),
                cb.reshape(-1), ignore_index=-100)

            top5 = out.codebook_logits.reshape(-1, out.codebook_logits.size(-1)).topk(5, dim=-1).indices
            alvo_flat = cb.reshape(-1, 1)
            acerto = (top5 == alvo_flat).any(dim=-1).float().mean()

            somas["base"] += base.item()
            somas["sem"] += sem.item()
            somas["top5"] += acerto.item()
            n_ok += 1

        if n_ok == 0:
            print(f"{nome:6s} sem amostra valida")
            continue
        r = {k: v / n_ok for k, v in somas.items()}
        r["loss"] = r["base"] + r["sem"]
        r["n"] = n_ok
        resultados[nome] = r
        print(f"{nome:6s} loss {r['loss']:7.3f}  base {r['base']:7.3f}  "
              f"semantic {r['sem']:7.3f}  top5 {r['top5']:.4f}  (n={n_ok})")

    import math
    print(f"\nacaso: base = ln(vocab) = {math.log(modelo.config.vocab_size):.3f}   "
          f"semantic = ln({modelo.config.codebook_size}) = "
          f"{math.log(modelo.config.codebook_size):.3f}")

    destino = Path(args.out) if args.out else ROOT / "reports" / "probe_format.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(
        {"ckpt": args.ckpt, "lora_ckpt": args.lora_ckpt, "protos": args.protos,
         "n": args.n, "seed": args.seed, "resultados": resultados},
        indent=2), encoding="utf-8")
    print(f"\n-> {destino}")


if __name__ == "__main__":
    main()
