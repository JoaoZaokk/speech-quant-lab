# Dois bugs no fine-tune do Fish Speech S2 Pro

Ambos são descasamentos entre o caminho de **treino** e o de **inferência**.
Ambos aparecem como "o modelo aprendeu mas a geração está ruim", que é o
sintoma mais caro de diagnosticar porque a `val/loss` continua bonita.

> **Built with Fish Audio.** Isto é relato de uso do
> [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro), não crítica ao
> modelo. O modelo é excelente; o caminho de fine-tune é que tem essas duas
> arestas.

---

## Bug 1 — o formato de treino não é o formato de inferência

O dataset que acompanha o repositório monta a sequência de um jeito, e a
inferência monta de outro:

```
TREINO   (AutoTextSemanticInstructionIterableDataset.pack_sentences)
    Speak out the provided text.
    <|speaker:user|> {texto}<|im_end|>
    <|speaker:assistant|> <|voice|>{códigos VQ}<|im_end|>

INFERÊNCIA   (text2semantic/inference.py, com referência)
    <|im_start|>system
    convert the provided text to speech reference to the following:

    Text:
    <|speaker:0|>{texto da referência}

    Speech:
    {códigos VQ da referência}<|im_end|>
    <|im_start|>user
    {texto alvo}<|im_end|>
    <|im_start|>assistant
    <|voice|>{gera}
```

Papéis com `<|im_start|>` contra strings literais `<|speaker:user|>`, texto de
sistema diferente, e — o principal — **a inferência sempre tem áudio de
referência e o treino nunca tem**.

### Como medir sem treinar nada

`probe_format.py` roda o modelo **cru** nos três formatos e compara a perda.
n = 64:

| formato | loss | componente texto | top5 |
|---|---:|---:|---:|
| `atual` (dataset do repo) | **29,70** | 12,80 | 0,0052 |
| `chat` (papéis, sem referência) | 14,14 | 6,69 | 0,0430 |
| `ref` (formato da inferência) | **13,70** | **6,33** | 0,0435 |
| *acaso, ln(155776)* | — | *11,96* | — |

O formato do repositório fica **acima do acaso**. Um modelo pré-treinado não
fica no acaso na própria tarefa — isso é entrada fora da distribuição, não
falta de fine-tune.

```bash
python fixes/probe_format.py --ckpt /caminho/para/s2-pro --protos /caminho/para/protos
```

### A correção

`semantic_ref.py` — `ReferenceConditionedIterableDataset`. Produz amostras no
formato da inferência, com um clipe de referência do mesmo locutor em
`prob_ref` das amostras (usamos 0,9). Amostra que estoura `max_length` é
descartada, nunca truncada — truncar a referência recria o problema.

Copie para `fish_speech/datasets/semantic_ref.py` e aponte o alvo do Hydra:

```
train_dataset._target_=fish_speech.datasets.semantic_ref.ReferenceConditionedIterableDataset
+train_dataset.prob_ref=0.9
```

Efeito, mesmo pack e mesma LoRA, trocando só o formato:

| | similaridade | rolloff | decaimento |
|---|---|---|---|
| base | 0,9598 | 4369 Hz | −5,4 dB |
| formato quebrado | 0,9340 | 3773 Hz | **−22,4 dB** |
| formato corrigido | 0,9600 | 4018 Hz | **−5,8 dB** |

---

## Bug 2 — embedding semântico escalado só na inferência

**Não fomos nós que achamos.** Crédito de
[`notmax123/Fish-Audio-S2-Pro-He`](https://huggingface.co/notmax123/Fish-Audio-S2-Pro-He),
que encontrou treinando hebraico, depois de cinco runs colapsarem. Está aqui
porque confirmamos no código e porque muda a leitura do nosso próprio
resultado.

Em `fish_speech/models/text2semantic/llama.py`:

```python
# embed()  — caminho de TREINO
x = self.embeddings(inp[:, 0]) + vq_embeds_sum
return x                                    # sem escala nenhuma

# forward_generate()  — caminho de INFERÊNCIA
if self.config.scale_codebook_embeddings:
    x = torch.where(vq_masks_expanded,
                    x / math.sqrt(self.config.num_codebooks + 1), x)
```

O `s2-pro` traz `scale_codebook_embeddings: true` e `num_codebooks: 10`, então
a inferência divide as posições semânticas por √11 ≈ 3,3166 e o treino não.
**Todo fine-tune treina contra embeddings 3,32× maiores do que os que verá ao
gerar.**

Sintoma: perda teacher-forced boa, geração colapsando depois das primeiras
palavras. `notmax123` liga isso a
[#1136](https://github.com/fishaudio/fish-speech/issues/1136),
[#682](https://github.com/fishaudio/fish-speech/issues/682) e
[#814](https://github.com/fishaudio/fish-speech/issues/814).

Números dele, antes e depois de corrigir:

| | RMS da amostra | decaimento de energia |
|---|---|---|
| antes | 0,008 – 0,022 | 0,07× (≈ −23 dB) |
| depois | 0,171 – 0,205 | 1,02× |
| base | 0,181 | — |

### Por que isso importa mesmo para quem treinou só o Fast AR

O Fast AR fica **depois** da pilha Slow e recebe `hidden_out`, nunca o embedding
cru — então é largamente imune. LoRA no Slow AR come a escala errada na entrada
e quebra. Foi exatamente o padrão que observamos e não conseguíamos explicar:
adaptador no Fast AR mede bem, adaptador no Slow AR dá −25 dB de decaimento.

**Os adaptadores que publicamos foram treinados com este bug presente.** Medem
bem por imunidade, não por correção. Se você vai treinar o Slow AR, corrija
antes — e meça **decaimento de energia**, não `val/loss`, que ranqueia ao
contrário.

### A correção, e o que ela mede

```python
# embed()  — caminho de TREINO, corrigido
x = self.embeddings(inp[:, 0]) + vq_embeds_sum

if self.config.scale_codebook_embeddings:
    mascara = is_semantic.unsqueeze(-1).expand_as(x)
    x = torch.where(mascara, x / math.sqrt(self.config.num_codebooks + 1), x)
return x
```

Medido no s2-pro **cru**, sem treinar nada, com `probe_format.py` (n=64,
semente 7). A versão sem escala fica atrás de um gate por variável de ambiente,
para o A/B ser o mesmo binário:

| formato | | loss | texto | semântico | top5 |
|---|---|---:|---:|---:|---:|
| `atual` | sem escala | 29,393 | 12,608 | 16,784 | 0,0057 |
| | com escala | 29,268 | 12,467 | 16,801 | 0,0033 |
| `ref` (o de treino) | sem escala | 13,663 | 6,352 | 7,311 | 0,0434 |
| | **com escala** | **11,571** | **4,844** | 6,727 | **0,1185** |

A run sem escala reproduz o 29,70 / 13,70 do Bug 1 — o controle é o mesmo, não
é medida nova de outra coisa. No formato `ref` a escala derruba **2,09** de
perda e quase **triplica o top5**. O formato `atual` mal se mexe: ele está fora
da distribuição por outro motivo, e esta correção não o salva.

---

## Licença

`semantic_ref.py` é uma modificação do código do fish-speech e segue a licença
do projeto original. `probe_format.py` é original e está sob a licença deste
repositório. Ver [`../LICENSE`](../LICENSE) e [`../LICENSES.md`](../LICENSES.md).
