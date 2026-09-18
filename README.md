# Speech Quant Lab

**Training, quantization, acceleration and evaluation for TTS/STT models.**

Por **[JoaoZaokk](https://github.com/JoaoZaokk)** — laboratório, curadoria,
direção e as decisões de escuta.

Bancada aberta. O que está aqui é o que serve para outra pessoa: **os números
com o método que os produziu**, o código que roda os benchmarks, e o que deu
errado — inclusive as medições minhas que estavam furadas e custaram horas.

Três modelos passaram por aqui:

| modelo | o que foi feito | estado |
|---|---|---|
| **[Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro)** | LoRA PT-BR, dois bugs de fine-tune, 5x no passo de treino | o mais coberto |
| **LFM** | LoRA PT-BR — CER 0,4234 → 0,0192 em 7 h | treina e aprende; `torch.compile` **piora** aqui |
| **VoxCPM** | avaliado contra os outros | sem veredito — o "ganho" que eu media era sorteio de voz |

A parte de aceleração e quantização **não é específica de TTS**: o perfil de
kernel, o INT8, o bucket por comprimento e as armadilhas de `torch.compile`
valem para qualquer transformer com sequência curta.

**O que fica de fora, e por quê:** áudio, manifests, pesos e checkpoints. O
áudio de treino inclui TAGARELA (CC BY-NC-SA 4.0), que não pode ser
redistribuído daqui, e os manifests carregam as transcrições dele. Código,
benchmark e resultado ficam todos aqui.

> **Built with Fish Audio.**
> O modelo base, o codec, o tokenizer e toda a capacidade multilíngue são da
> Fish Audio / 39 AI, INC. Aqui foram treinados 7,1 M de parâmetros em cima
> disso. A proporção importa e vai aparecer repetida.

## Resultado

**→ [`JoaoZaokk/fish-s2-pro-ptbr-lora`](https://huggingface.co/JoaoZaokk/fish-s2-pro-ptbr-lora)**
— três adaptadores LoRA, sem os pesos base. O recomendado é o `ptbr-2h-256spk-scalefix`.

Medido em 216 frases, 24 locutores do split `dev` do Common Voice, disjunto de
todo pack de treino:

| | base (Fish Audio) | 2 h / 46 loc | 2 h / 256 loc | 2 h / 256 loc **+ fix** |
|---|---|---|---|---|
| CER ↓ | 0,0220 | **0,0180** | 0,0195 | 0,0183 |
| WER ↓ | 0,0486 | **0,0404** | 0,0472 | 0,0429 |
| similaridade ↑ | 0,9577 | 0,9566 | 0,9574 | 0,9571 |
| decaimento ↑ | −5,70 dB | −5,68 dB | −5,81 dB | **−5,32 dB** |

Leitura honesta: **ganho pequeno em inteligibilidade, empate no resto.** Dois
testes cegos pareados contra o base deram 6–6 e 4–5 (p = 1,000 nos dois). O que
os adaptadores comprovadamente **não** fazem é estragar o modelo base — e essa
foi a parte que deu trabalho.

### O que a correção do bug de escala rendeu

O `+ fix` é o mesmo experimento do `2 h / 256 loc` rodado de novo, com uma
divisão a mais no `embed()`. Mesmo pack, mesma LoRA, mesmos 3000 passos, mesmo
`lr`. Pareado item a item nas 216 frases:

| métrica | 256 loc | + fix | delta | fix melhor em | p |
|---|---:|---:|---:|---:|---:|
| **decaimento** | −6,1264 dB | **−5,4809 dB** | **+0,65 dB** | 129/216 | **0,0052** |
| CER | 0,0195 | 0,0183 | −0,0012 | 24 × 13 (179 empates) | 0,099 |
| similaridade | 0,9548 | 0,9543 | −0,0005 | 97/215 | 0,17 |
| rolloff | 4337,6 Hz | 4270,7 Hz | −67,0 Hz | 99/216 | 0,25 |

Quatro testes na mesma família; com Bonferroni o limiar vira 0,0125 e o
decaimento passa mesmo assim. O CER inclina a favor mas não fecha.

**A correção devolve fôlego no fim da frase e não cobra nada.** O tamanho também
é o previsto: o Fast AR fica *depois* da pilha slow e nunca vê o embedding cru,
então é largamente imune. São +0,65 dB aqui contra os ~+23 dB que o `notmax123`
mediu corrigindo no Slow AR.

Uma coisa que o número não pega, vinda da escuta: **com o fix, um `[whisper]` no
início da frase produz sussurro de verdade; sem ele, não.** É uma observação
única, não uma medida — mas é consistente com o mecanismo, porque a posição 0 é
onde a escala errada mais distorce o condicionamento inicial.

## Os dois bugs

Estão em [`fixes/`](fixes/), com reprodução e correção.

1. **O formato de treino do repositório não é o formato de inferência.** Dá
   para provar sem treinar nada: o modelo cru pontua **29,70** de perda no
   formato do dataset de fine-tune, contra 13,70 no formato da inferência — e
   o acaso é 11,96. Treinar assim ensina um prompt que o modelo nunca vai
   receber.

2. **Embeddings semânticos são divididos por √11 na inferência e não no
   treino.** Todo fine-tune treina contra embeddings 3,32× maiores do que os da
   geração. Achado por
   [`notmax123`](https://huggingface.co/notmax123/Fish-Audio-S2-Pro-He), não por
   nós. **Confirmado por medida no modelo cru:** aplicar a escala derruba a
   perda no formato de treino de 13,663 para 11,571 e leva o top5 de 0,0434 a
   0,1185 — quase o triplo. Detalhe em [`fixes/README.md`](fixes/README.md).

## O que a gente leu errado

A parte mais útil destas notas. O erro dominante não foi métrica quebrada — foi
**leitura errada de métrica que funcionava**:

| métrica | eu li como | mede de fato |
|---|---|---|
| `val/loss` | qualidade | dano no Slow AR — a menor perda foi o pior de ouvir |
| similaridade | clonagem | memorização (as vozes de teste estavam no treino) |
| duração | corte | nada — a fala continua, só fica baixa |
| CER | pronúncia errada | o ASR não **ouviu** (volume, não fonema) |
| rolloff | virou chiado | piso de ruído do arquivo, em sinal a −60 dB |
| NISQA | qualidade percebida | degradação de canal; pune expressividade |

Regra que sobrou: quando a métrica e o ouvido divergem, a pergunta é **"o que
essa régua lê fisicamente?"** — nunca "como convenço de que o número está
certo".

A que sobreviveu: **decaimento**, o nível do último quarto menos o do primeiro,
em dB, com o silêncio das pontas aparado. Foi a única que apontou o mesmo
defeito em pt, en e es, e a única que casou com o relato de ouvido em todos os
casos.

## Três coisas que custaram tempo e valem contar

### O Slow AR aprende mais e quebra

A LoRA padrão do repositório alcança o Slow AR. Ela produz a **melhor
`val/loss` de todas** (9,82 contra 12,11) e é o pior modelo de ouvir: −25 dB de
decaimento, 8 dB abaixo em nível, CER 3,4× o base.

O defeito não é corte nem problema de idioma. É **envelope** — o modelo fala o
texto inteiro e some enquanto fala. Amplificando a cauda 34 dB a frase completa
aparece. Três métricas independentes (duração, CER, rolloff) leram isso como
"cortou", e as três estavam erradas. Por isso os adaptadores publicados
congelam o Slow AR.

### A escada de treino não media o que dizia medir

A primeira escada usava **os mesmos 136 grupos de voz em todos os degraus**, de
30 min a 30 h. Subir a escada aumentava minutos por locutor, nunca o número de
locutores. A pergunta "mais dado ajuda?" nunca chegou a ser feita — o que se
testou foi "mais repetição do mesmo locutor ajuda?".

A correção põe teto de 6 min por grupo, o que baixa o total de 31,9 h para
16,6 h. Troca deliberada: variedade acima de volume.

### Referência descasada mede a referência

Uma versão do teste multilíngue gerou inglês e espanhol com referência **em
português** e concluiu que o modelo tinha sotaque. O sotaque era da referência.
Refeito com referência no idioma do texto, 40 gerações:

| checkpoint | duração | decaimento | nível |
|---|---|---|---|
| base (Fish Audio) | 13,17 s | −1,3 dB | −25,2 dBFS |
| 2 h / 46 loc | 12,88 s | −2,2 dB | −25,2 dBFS |
| 2 h / 256 loc | 13,12 s | −1,1 dB | −25,1 dBFS |
| *(um run no Slow AR, contraste)* | *11,14 s* | *−25,3 dB* | *−35,5 dBFS* |

Adaptar para português **não** estragou inglês nem espanhol.

## Tags de emoção — parcial

| tag | efeito medido |
|---|---|
| `[sigh]`, `[exhale]`, `[short pause]`, `[emphasis]` | funciona — tempo e respiração respondem |
| `[whisper]` | funciona — −2,2 dB |
| `[laughing]`, `[chuckle]` repetidas | duração escala monotonicamente com a repetição |
| maioria das tags de volume | sem efeito mensurável |

### O controle com tags inventadas — fechado

A dúvida era se o modelo interpreta a tag ou apenas conta colchetes. Escada
inteira regerada numa condição só, 13 prompts × 3 sementes, depois 5 condições
× 8 sementes em n=4, tudo transcrito com Whisper large-v3.

**Nenhuma palavra placebo é pronunciada** em nenhuma amostra.

| condição | duração | vs controle | p |
|---|---:|---:|---:|
| controle | 6,44 ± 0,25 s | — | — |
| placebo (`[glorp]`, `[zibbe]`), n=16 | 7,17 ± 0,53 s | +0,73 s | 0,0019 |
| tag real (`[laughing]`, `[chuckle]`), n=16 | 8,09 ± 0,58 s | +1,65 s | <1e-5 |
| real vs placebo | — | +0,92 s | 0,00006 |

**44% do alongamento acontece com uma tag inventada.** O colchete sozinho já
faz alguma coisa; a tag real faz mais em cima disso. As duas hipóteses estavam
meio certas.

Com 3 sementes os placebos pareciam planos (+0,16 e +0,35 s) — só em n=4 com 8
sementes a diferença apareceu. Fica o registro de que a leitura de 3 sementes
teria virado uma afirmação errada aqui.

**Um efeito que replicou:** referência gravada **com** colchetes na própria
transcrição gera saída mais longa que a mesma referência sem — 13 de 15 pares,
teste de sinal p = 0,0074, média 12,70 s → 13,41 s (+5,6%). O texto gerado não
tem tag em nenhum dos lados. Se é expressividade ou só fala mais devagar, ainda
não está resolvido.

### O que estes testes NÃO cobrem

Tudo acima usa **tag de palavra solta** (`[whisper]`, `[laughing]`, `[sad]`).
O model card oficial do S2 Pro diz outra coisa:

> "Rather than relying on a fixed set of predefined tags, S2 Pro accepts
> **free-form textual descriptions** — such as `[whisper in small voice]`,
> `[professional broadcast tone]`, or `[pitch up]`"
> — "15,000+ unique tags supported"

Palavra solta também está documentada, então não é forma inválida. Mas a
**descrição livre não foi testada aqui**, e é plausível que uma frase condicione
melhor por carregar mais sinal.

Leitura honesta: estes números medem que **tag de palavra solta é pouco
confiável**. Não medem que o controle por tag não funcione.

## Receita dos adaptadores

```
LoRA        r_32_alpha_16_fast   (r=32, alpha=16, alpha/r = 0,5)
alvos       fast_embeddings, fast_layers.{0..3}.attention.{wqkv,wo},
            fast_layers.{0..3}.feed_forward.{w1,w2,w3}, fast_output
congelado   as 36 camadas do Slow AR, o codec, o tokenizer
otimizador  lr 1e-5, bf16-true
batch       1 x 8 de acumulação (efetivo 8)
max_length  1024
passos      3000, validação a cada 100
dataset     condicionado a referência, prob_ref = 0,9
hardware    1x RTX 3090, ~7 h por run
```

Dados: 2 h de pt-BR em duas distribuições de voz diferentes — 46 e 256
locutores do Common Voice, mais clipes do TAGARELA. Detalhe por fonte no card
do Hugging Face.

## O que está aqui

```
README.md      estas notas
notes/         o que foi medido: texto, desempenho, aceleração
fixes/         bugs de fine-tune, com reprodução e correção
bench/         os scripts que produziram cada número
accel/         GEMM INT8 para os lineares congelados
reports/       a saída crua dos benchmarks, em JSON
LICENSE        MIT, para o código deste repositório
LICENSES.md    as licenças das fontes usadas
```

Os benchmarks acham o checkpoint por `$FISH_CKPT` ou `--ckpt`; não há caminho
de máquina em lugar nenhum.

- **[`notes/simbolos-e-normalizacao.md`](notes/simbolos-e-normalizacao.md)** —
  o que cada símbolo produz no s2-pro, medido, com a saída literal de cada caso.
  Qual entrada quebra, o que ela vira, e qual forma escrita o modelo lê certo.
  Inclui as regras que foram para o lixo assim que alguém escutou.
- **[`notes/desempenho-upstream.md`](notes/desempenho-upstream.md)** — três
  patches de terceiros, abertos e não mergeados, que aceleram o S2 Pro e cortam
  VRAM. Crédito dos autores; aqui é só o mapa e o motivo de `git log` não
  mostrar.
- **[`notes/acelerar-o-treino.md`](notes/acelerar-o-treino.md)** — 5x no passo
  de treino numa RTX 3090, medido braço a braço: onde o tempo está de verdade
  (88,7% dos kernels duram menos de 10 µs), por que `torch._int_mm` é mais lento
  que bf16, por que uma `autograd.Function` em Python anula o ganho sob
  `torch.compile`, e por que ordenar o lote por comprimento vale mais que
  packing. Inclui o que **não** funcionou e os quatro erros de medição que quase
  viraram número publicado.

- **[`fixes/lora-slow-ar.md`](fixes/lora-slow-ar.md)** — dois bugs que fazem o
  LoRA treinar a metade errada do DualAR: `setup_lora` não tem prefixo `slow_*`
  (só `fast_*`), então "treinar só o Slow" é impossível sem patch; e
  `use_reentrant=True` mata o treino assim que o LoRA é parcial, com uma
  mensagem que não aponta para nenhum dos dois. Compostos, deram **p = 1** num
  teste de notação — eu estava treinando o subsistema que não lê texto.
- **[`bench/`](bench)** — `bench_int8_kernel.py` (GEMM isolado por forma),
  `bench_treino_fish.py` (passo de treino, um acelerador por vez),
  `perfil_passo_fish.py` (`torch.profiler`, kernel a kernel),
  `erro_int8_fish.py` (erro nos logits e no gradiente da LoRA),
  `medir_padding_fish.py` (quanto do lote é padding e quanto o bucket salva),
  `matriz_aceleradores.py` (kernel × compile × batch, um processo por célula).
- **[`accel/int8_linear.py`](accel/int8_linear.py)** — troca os `nn.Linear`
  **congelados** por GEMM INT8, mantendo `weight` no `state_dict` com o mesmo
  nome. O backward vai em bf16 sobre o peso original, o que o torna exato, e o
  autograd é registrado **no op** — não numa `autograd.Function`, pelo motivo
  medido na nota.

**O que não está, de propósito:** áudio, pesos, manifests, checkpoints e os
scripts de aquisição e curadoria. O áudio de treino inclui TAGARELA, que é
CC BY-NC-SA 4.0 e não pode ser redistribuído daqui, e os manifests carregam as
transcrições dele.

## Licenças

| item | licença |
|---|---|
| código deste repositório | MIT |
| `fishaudio/s2-pro` e qualquer LoRA derivado | **Fish Audio Research License** — pesquisa e uso não comercial |
| TAGARELA | CC BY-NC-SA 4.0 — não comercial |
| Common Voice PT | CC0 / permissiva |

Os adaptadores publicados são **não comerciais por dois caminhos
independentes**: a licença da Fish Audio no modelo base, e a CC BY-NC-SA do
TAGARELA nos dados. Uso comercial dos materiais da Fish Audio ou de qualquer
obra derivada exige acordo escrito separado com a Fish Audio
(business@fish.audio).

## Autoria

**[JoaoZaokk](https://github.com/JoaoZaokk)** — autor do trabalho. Montou o
laboratório, curou os dados, dirigiu os experimentos e decidiu o rumo em cada
bifurcação.

Vale destacar uma contribuição específica, porque é a que mais mudou o
resultado: **as correções vieram do ouvido dele, não das métricas.** Sete vezes
neste projeto uma medida disse "está tudo igual" e a escuta disse "não está" —
e a escuta estava certa nas sete. O corte no fim da frase, o inglês com sotaque
que era a referência e não o modelo, o checkpoint de menor perda que era o pior
de ouvir: nenhum desses foi achado por um número. Vários dos métodos descritos
aqui — `decaimento_db`, o banco de vozes limpo, a regra de não escolher
checkpoint por `val/loss` — existem porque uma métrica falhou primeiro e alguém
percebeu ouvindo.

A coleta de vozes sintéticas (103 personagens, avaliados um a um de ouvido, com
a nota escrita no nome da pasta) também é dele.

## Créditos

**Fish Audio / 39 AI, INC.** — pelo S2 Pro. Para ser claro sobre a divisão de
trabalho: eles treinaram um modelo de fala multilíngue de 4,6 bilhões de
parâmetros, alinharam com RL, construíram o codec e o tokenizer, e liberaram
sob uma licença que permite exatamente este tipo de trabalho de graça. Aqui
foram treinados 7,1 milhões de parâmetros em duas horas de áudio de podcast
numa GPU de consumidor, e boa parte do que estas notas contribuem é uma lista
de formas de ler mal a própria medição. **Built with Fish Audio**, e dá pra
notar.

- **freds0** — dataset TAGARELA e o artigo do ICASSP 2026.
- **Mozilla Common Voice** — a única parte dos dados de treino que pode ser
  redistribuída livremente.
- **notmax123** — achou o bug de escala do embedding e publicou em vez de
  corrigir calado.
- **TachibanaKimika** — o repositório de adaptadores que serviu de modelo para
  o tratamento de licença e a fixação de revisão do base.
- **neko-legends** — por publicar um perfil de serviço sem pesos dentro.
