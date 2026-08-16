"""PATCH LOCAL — dataset no formato que a INFERENCIA usa.

POR QUE ISTO EXISTE
-------------------
`AutoTextSemanticInstructionIterableDataset` (o dataset padrao do repo) monta a
sequencia de um jeito, e `models/text2semantic/inference.py` monta de outro.
Medido no s2-pro CRU, 64 amostras do gold_30m (scripts/probe_format.py):

    formato                          loss    base    semantic   top5
    ---------------------------------------------------------------
    atual   (dataset do repo)       29.70   12.80      16.90   0.0052
    chat    (papeis, sem ref)       14.14    6.69       7.44   0.0430
    ref     (igual a' inferencia)   13.70    6.33       7.37   0.0435

    acaso:  base = ln(155776) = 11.96      semantic = ln(4096) = 8.32

O formato do repo esta' ACIMA do acaso nos dois componentes. Nao e' "falta
treinar": e' entrada que o checkpoint nunca viu. Treinar assim ensina o modelo
um prompt que ele nao vai receber na hora de gerar — e foi isso que produziu a
degradacao que a gente vinha perseguindo (fala que morre no fim da frase,
clonagem pior, banda alta perdida).

O QUE MUDA
----------
Cada amostra vira exatamente o que `generate_long` monta:

    <|im_start|>system
    convert the provided text to speech reference to the following:

    Text:
    <|speaker:0|>{texto da referencia}

    Speech:
    {codigos VQ da referencia}<|im_end|>
    <|im_start|>user
    {texto alvo}<|im_end|>
    <|im_start|>assistant
    <|voice|>{codigos VQ do alvo}<|im_end|>

Perda so' na fala do assistente. A referencia entra com `cal_loss=False`, entao
o `semantic_mask` do `lit_module` a ignora — os rotulos dela ficam em -100.

Grupo com um clipe so' nao tem referencia possivel do mesmo locutor; esses caem
no formato sem referencia (`convert the provided text to speech`), que a
inferencia tambem usa quando ninguem passa audio de exemplo.
"""

import random
from pathlib import Path
from random import Random
from typing import Optional

import torch
from torch.utils.data import IterableDataset

from fish_speech.content_sequence import TextPart, VQPart
from fish_speech.conversation import Conversation, Message
from fish_speech.datasets.protos.text_data_stream import read_pb_stream
from fish_speech.datasets.semantic import split_by_rank_worker
from fish_speech.text.clean import clean_text
from fish_speech.tokenizer import FishTokenizer
from fish_speech.utils import RankedLogger
from fish_speech.utils.braceexpand import braceexpand

log = RankedLogger(__name__, rank_zero_only=True)

CODEBOOK_PAD_TOKEN_ID = 0

SISTEMA_COM_REF = (
    "convert the provided text to speech reference to the following:\n\nText:\n"
)
SISTEMA_SEM_REF = "convert the provided text to speech"


class ReferenceConditionedIterableDataset(IterableDataset):
    """Amostras no formato da inferencia, com audio de referencia do locutor.

    Args:
        proto_files: os `.protos` do build_dataset.py
        tokenizer: FishTokenizer
        prob_ref: chance de usar referencia quando o grupo tem >=2 clipes.
            Nao e' 1.0 de proposito: a inferencia tambem roda sem referencia, e
            deixar uma fracao sem exemplo mantem esse caminho vivo.
        max_ref: quantos clipes de referencia concatenar (a inferencia aceita
            varios). 1 e' o caso comum.
        seed: usada so' para ordenar/embaralhar os arquivos, como no repo.
    """

    def __init__(
        self,
        proto_files: list[str],
        tokenizer: FishTokenizer = None,
        prob_ref: float = 0.9,
        max_ref: int = 1,
        max_length: int = 1024,
        num_codebooks: Optional[int] = None,
        seed: int = 42,
        # aceitos e ignorados: o yaml do repo passa isto para o dataset padrao e
        # falhar aqui por causa de um kwarg herdado nao ajuda ninguem
        causal: bool = True,
        use_speaker: bool = False,
        interactive_prob: float = 0.0,
        skip_text_prob: float = 0.0,
    ):
        super().__init__()
        assert 0.0 <= prob_ref <= 1.0, "prob_ref precisa estar em [0, 1]"

        self.proto_files = proto_files
        self.tokenizer = tokenizer
        self.prob_ref = prob_ref
        self.max_ref = max_ref
        self.max_length = max_length
        self.num_codebooks = num_codebooks
        self.seed = seed
        self.groups = None

    # ------------------------------------------------------------------ dados

    def _carregar(self):
        if self.groups is not None:
            return

        expandidos = []
        for filename in self.proto_files:
            for i in braceexpand(filename):
                i = Path(i)
                if i.is_file():
                    expandidos.append(i)
                elif i.is_dir():
                    expandidos.extend(i.rglob("*.proto"))
                    expandidos.extend(i.rglob("*.protos"))
                else:
                    raise ValueError(f"{i} is not a file or directory")

        expandidos = sorted(expandidos)
        Random(self.seed).shuffle(expandidos)

        self.groups = []
        for filename in split_by_rank_worker(expandidos):
            with open(filename, "rb") as f:
                for text_data in read_pb_stream(f):
                    frases = [
                        s for s in text_data.sentences if s.texts and s.semantics
                    ]
                    if frases:
                        self.groups.append(frases)

        Random(self.seed).shuffle(self.groups)
        self.group_weights = [len(g) for g in self.groups]
        com_par = sum(1 for g in self.groups if len(g) >= 2)
        log.info(
            f"Read {len(self.groups)} groups "
            f"({com_par} with >=2 clips, usable as reference)"
        )

    def __iter__(self):
        while True:
            item = self.montar()
            if item is not None:
                yield item

    # ---------------------------------------------------------------- formato

    def _codes(self, frase) -> torch.Tensor:
        return torch.tensor([x.values for x in frase.semantics]).to(torch.int32)

    def montar(self):
        self._carregar()

        grupo = random.choices(self.groups, weights=self.group_weights, k=1)[0]

        usa_ref = len(grupo) >= 2 and random.random() < self.prob_ref
        if usa_ref:
            k = min(self.max_ref + 1, len(grupo))
            escolhidos = random.sample(range(len(grupo)), k)
            alvo = grupo[escolhidos[0]]
            refs = [grupo[i] for i in escolhidos[1:]]
        else:
            alvo = random.choice(grupo)
            refs = []

        if refs:
            partes = [TextPart(text=SISTEMA_COM_REF)]
            texto_ref = "\n".join(
                f"<|speaker:{i}|>{clean_text(r.texts[0])}" for i, r in enumerate(refs)
            )
            partes.append(TextPart(text=texto_ref))
            partes.append(TextPart(text="\n\nSpeech:\n"))
            partes.append(
                VQPart(codes=torch.cat([self._codes(r) for r in refs], dim=1),
                       cal_loss=False)
            )
        else:
            partes = [TextPart(text=SISTEMA_SEM_REF)]

        conv = Conversation(
            [
                Message(role="system", parts=partes, cal_loss=False),
                Message(
                    role="user",
                    parts=[TextPart(text=clean_text(alvo.texts[0]))],
                    cal_loss=False,
                ),
                Message(
                    role="assistant",
                    parts=[VQPart(codes=self._codes(alvo), cal_loss=True)],
                    cal_loss=True,
                    modality="voice",
                ),
            ]
        )

        # `Conversation.encode` repassa `max_length=` para `ContentSequence.encode`,
        # que nao aceita esse parametro nesta versao — TypeError na hora. O
        # `to_content_sequence()` monta as mesmas partes e desvia do bug.
        encoded = conv.to_content_sequence().encode(tokenizer=self.tokenizer)

        num_codebooks = (
            len(alvo.semantics) if self.num_codebooks is None else self.num_codebooks
        )

        tokens_raw = encoded.tokens
        tokens = torch.zeros((num_codebooks + 1, len(tokens_raw)), dtype=torch.int)
        tokens[0] = tokens_raw

        vq_parts = [p.to(tokens.device) for p in encoded.vq_parts]
        vq_parts = torch.cat(vq_parts, dim=1)
        tokens[1:, encoded.vq_mask_tokens] = vq_parts

        labels_raw = encoded.labels
        labels = torch.full((num_codebooks + 1, len(labels_raw)), -100, dtype=torch.int)
        labels[0, :] = labels_raw
        labels[1:, encoded.vq_mask_labels] = vq_parts
        labels[1:, -1:] = CODEBOOK_PAD_TOKEN_ID

        tokens = tokens.long()
        labels = labels.long()

        assert (tokens[1:, ~(encoded.vq_mask_tokens)] == CODEBOOK_PAD_TOKEN_ID).all()
        assert (labels[1:, -1:] == CODEBOOK_PAD_TOKEN_ID).all()
        assert tokens.size(1) == labels.size(1), f"{tokens.size(1)} != {labels.size(1)}"

        # A referencia entra INTEIRA ou nao entra: cortada no meio pelo collator,
        # ela vira ruido no system prompt e o alvo some do fim da sequencia.
        if tokens.size(1) > self.max_length:
            return None

        return {"tokens": tokens, "labels": labels}
