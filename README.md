# PT-BR Audio Lab

Notas de um estudo de adaptação de TTS ao **português brasileiro**, usando
**[Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro)**.

Isto é a **escrita**, não o laboratório. O que está aqui é o que serve para
outra pessoa: o que foi medido, o que deu errado, e os dois bugs de fine-tune
com reprodução. O pipeline de dados, os scripts internos e os manifests ficam
fora — dependem da máquina onde rodam e não ajudam ninguém.

> **Built with Fish Audio.**
> O modelo base, o codec, o tokenizer e toda a capacidade multilíngue são da
> Fish Audio / 39 AI, INC. Aqui foram treinados 7,1 M de parâmetros em cima
> disso. A proporção importa e vai aparecer repetida.

## Resultado

**→ [`JoaoZaokk/fish-s2-pro-ptbr-lora`](https://huggingface.co/JoaoZaokk/fish-s2-pro-ptbr-lora)**
— dois adaptadores LoRA, sem os pesos base.

Medido em 216 frases, 24 locutores do split `dev` do Common Voice, disjunto de
todo pack de treino:

| | base (Fish Audio) | 2 h / 46 loc | 2 h / 256 loc |
|---|---|---|---|
| CER ↓ | 0,0220 | **0,0180** | 0,0195 |
| WER ↓ | 0,0486 | **0,0404** | 0,0472 |
| similaridade ↑ | 0,9577 | 0,9566 | 0,9574 |
| decaimento | −5,70 dB | −5,68 dB | −5,81 dB |

Leitura honesta: **ganho pequeno em inteligibilidade, empate no resto.** Dois
testes cegos pareados contra o base deram 6–6 e 4–5 (p = 1,000 nos dois). O que
os adaptadores comprovadamente **não** fazem é estragar o modelo base — e essa
foi a parte que deu trabalho.

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
   nós; confirmado aqui lendo as duas funções.

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

Um controle com tags inventadas (`[glorp]`, `[zibbe]`, `[thubner]`) em 1×/2×/4×
foi iniciado e **nunca analisado**. Ele decide se o modelo interpreta a tag ou
apenas conta colchetes. Enquanto isso não fechar, nenhuma conclusão sobre tags
está pronta.

**Um efeito que replicou:** referência gravada **com** colchetes na própria
transcrição gera saída mais longa que a mesma referência sem — 13 de 15 pares,
teste de sinal p = 0,0074, média 12,70 s → 13,41 s (+5,6%). O texto gerado não
tem tag em nenhum dos lados. Se é expressividade ou só fala mais devagar, ainda
não está resolvido.

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
fixes/         os dois bugs, com reprodução e correção
LICENSE        MIT, para o código deste repositório
LICENSES.md    as licenças das fontes usadas
```

**O que não está, de propósito:** áudio, pesos, manifests, os scripts de
aquisição e curadoria, e a configuração da máquina. O áudio de treino inclui
TAGARELA, que é CC BY-NC-SA 4.0 e não pode ser redistribuído daqui, e os
manifests carregam as transcrições dele.

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
