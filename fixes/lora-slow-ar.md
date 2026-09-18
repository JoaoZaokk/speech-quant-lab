<!-- Publicado em github.com/JoaoZaokk/speech-quant-lab -->

# Dois bugs que fazem o LoRA do S2 Pro treinar a metade errada do modelo

Achados em 2026-09-17/18 no
[`fishaudio/fish-speech`](https://github.com/fishaudio/fish-speech). Os dois se
manifestam **só** quando se tenta treinar um subconjunto do DualAR, e é
exatamente o que alguém faz ao querer ensinar leitura de notação.

O S2 Pro é DualAR: um **Slow AR** de 36 camadas que lê o TEXTO, e um **Fast AR**
de 4 camadas que expande cada token semântico em 10 codebooks acústicos. Quem
quer ensinar o modelo a ler `R$ 300.000,00` como "trezentos mil reais" precisa
mexer no Slow. Nos dois bugs abaixo, quem tenta isso acaba treinando outra coisa.

---

## 1. `setup_lora` não distingue Slow de Fast — só existe o prefixo `fast_*`

Em `fish_speech/models/text2semantic/lora.py`, os alvos são casados por nome, e
os nomes sem prefixo pegam **os dois transformers**:

```python
attention = "attention" in targets           # <- Slow E Fast
mlp       = "mlp" in targets                 # <- Slow E Fast
fast_attention = "attention" in targets or "fast_attention" in targets
fast_mlp       = "mlp" in targets or "fast_mlp" in targets
```

Existe `fast_attention` e `fast_mlp` para escolher só o Fast. **Não existe o
simétrico** — não há como pedir só o Slow. Um `target_modules: [attention, mlp]`
liga as 40 camadas, e o config de exemplo do repo é `fast`-only.

**Consequência medida:** treinei achando que estava no Slow, com um config que
casava `attention`/`mlp`. O merge depois mostrou no peso que o Slow nunca havia
sido tocado em um run, e que nos outros os dois estavam sendo treinados juntos.
Um teste de leitura de notação deu **p = 1** — o subsistema que lê texto estava
congelado o treino inteiro.

**Correção** — simétrico ao que já existe:

```python
slow_attention = "attention" in targets or "slow_attention" in targets
slow_mlp       = "mlp" in targets or "slow_mlp" in targets
fast_attention = "attention" in targets or "fast_attention" in targets
fast_mlp       = "mlp" in targets or "fast_mlp" in targets
```

Conferido contando os módulos LoRA depois do setup: com `slow_attention` +
`slow_mlp` saem 5 no Slow e 0 no Fast; com `attention` + `mlp`, 5 e 5.

---

## 2. `use_reentrant=True` mata o treino assim que o LoRA é parcial

Os dois laços de camada usam:

```python
x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
```

A versão reentrante do `torch.utils.checkpoint` só propaga gradiente se **algum
input do bloco** exigir grad. Parâmetro de dentro do bloco não conta.

Com LoRA apenas no Slow, a entrada do primeiro bloco vem do embedding
**congelado**. Nenhum input exige grad, a saída perde o `grad_fn`, e o treino
morre no primeiro passo:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**Por que ninguém esbarra nisso:** o config de exemplo (`fast`-only) inclui
`fast_embeddings` nos alvos. Com o embedding treinável, a entrada já tem grad e
a versão reentrante funciona. É coincidência da configuração de exemplo, não
desenho.

**Correção** — `use_reentrant=False`, que é a implementação recomendada pelo
PyTorch há várias versões e olha os parâmetros do módulo, não só os inputs:

```python
x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=False)
```

Serve para os dois casos; não há razão para manter o reentrante.

---

## Por que isso importa junto

Os dois compõem de um jeito ruim. O bug 1 faz "treinar só o Slow" ser
impossível sem patch. Quem contorna nomeando alvos manualmente esbarra no bug 2,
que **falha com uma mensagem que não aponta para nenhum dos dois** — fala de
tensor sem `grad_fn`, a 36 camadas de distância da causa.

Corrigidos os dois, com a receita de fine-tune completa da
[PR #1311](https://github.com/fishaudio/fish-speech/pull/1311) (warmup 100,
`bf16-mixed`, peso separado por modalidade na loss), a leitura de valores
monetários foi de **2/12 para 9/12** num conjunto de controle — o subsistema
certo passou a receber gradiente.

O custo: esquecimento em outras famílias de notação que quase não apareciam no
pack. Isso é composição de dataset, não hiperparâmetro — mas é outra nota.
