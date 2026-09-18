<!-- Publicado em github.com/JoaoZaokk/ptbr-audio-lab -->

# 5x no treino do S2 Pro numa RTX 3090 — e os quatro erros de medição no caminho

Levantado em 2026-09-18 treinando LoRA no
[`fishaudio/fish-speech`](https://github.com/fishaudio/fish-speech) S2 Pro
(DualAR, 36 camadas Slow + 4 Fast, `dim=2560`, `intermediate_size=9728`,
`vocab_size=155776`), numa RTX 3090 (sm86, 24 GB) com Python 3.13,
torch 2.13.0+cu130 e Triton no Windows.

Tudo marcado **[MEDIDO]** tem número e método. O que não tem, está marcado como
julgamento. Metade desta nota é sobre medições minhas que estavam erradas — elas
estão aqui porque cada uma custou horas e nenhuma era óbvia depois.

---

## Resumo

Comprimento mediano das amostras: **93 tokens semânticos**. Isso importa mais do
que parece — quase toda a conclusão abaixo se inverte em sequência longa.

```
                                        s/passo    amostras/s   ganho
base (batch 1, sem nada)                  0,223       4,48      1,00x
+ torch.compile                           0,182       5,49      1,23x
+ GEMM INT8 (forward)                     0,136       7,35      1,64x
+ batch 4                                 0,233      17,17      3,83x
+ batch 8                                 0,430      18,60      4,15x
+ gradient checkpointing OFF              0,354      22,60      5,04x
```

O que **não** ajudou, e por quê, está na seção 6. É a parte mais útil da nota.

---

## 1. Primeiro meça onde o tempo está, não onde você acha que está

Meu breakdown de FLOPs dizia: MLP 73%, projeções de atenção 26%, QK/AV 1,2%.
Conclusão natural: otimizar atenção é inútil, otimizar MLP é tudo.

O breakdown de **tempo** conta outra história. Com `torch.profiler`, filtrando
só kernel de GPU real:

```
familia            ms/passo      %
GEMM bf16            154,19    70,9%
elementwise           47,11    21,7%
otimizador             7,53     3,5%
reducao                4,63     2,1%
```

**11.290 kernels por passo.** Tempo médio *dentro* de um kernel: **19,3 µs**.
Buraco entre eles: **4,3 µs**. Distribuição por tamanho:

```
 µs por kernel   kernels   % dos kernels   % do tempo
          0-5      2.944         26,1%         4,2%
         5-10      7.069         62,6%        21,3%
        10-20        316          2,8%         2,3%
        20-50        363          3,2%         5,9%
       50-200        525          4,7%        34,8%
         >200         73          0,6%        31,4%
```

**88,7% dos kernels duram menos de 10 µs e consomem 25% do tempo.** Um
`cudaLaunchKernel` custa 5–10 µs de CPU, e antes dele vem o dispatcher do ATen,
o nó de autograd e a alocação da saída — em Python, com GIL. Pagava-se mais para
*pedir* do que para receber.

E lançamento de kernel é **fila serial**: núcleos extras não paralelizam uma
thread empurrando 11.290 kernels um a um.

### Armadilha do profiler

`key_averages()` devolve também wrappers de CPU (`aten::mm`, `MmBackward0`) cujo
`device_time_total` **já inclui** os kernels que eles lançam. Somar tudo conta o
mesmo trabalho duas vezes: na minha primeira versão deu **941 ms/passo** num
passo que o cronômetro media em 266 ms. Filtre por
`ev.device_type == DeviceType.CUDA` e use `self_device_time_total`.

---

## 2. `torch.compile` — 1,23x, e é o que destrava todo o resto

```
                kernels/passo   tempo de GPU   parede   GPU ociosa
sem compile            11.290       217,4 ms    266 ms   48,6 (18%)
com compile             3.982       174,3 ms    183 ms    8,7 ( 5%)
```

Corta **11.290 → 3.982 kernels** fundindo o elementwise (RMSNorm, SiLU, mul,
add, rotary) em kernels Triton. A GPU sai de 18% ociosa para 5% — praticamente
saturada.

Rende **mais em sequência curta** (1,77x em 160 tokens, 1,35x em 512), ao
contrário do que a intuição diz: com frase curta o gargalo é despacho, não FLOPs.

CUDA Graphs (`mode="reduce-overhead"`) rende +7,9% em batch 1 e só +2,2% em
batch 4 — batch maior, kernel maior, menos sensível a despacho. Não vale a
restrição de forma fixa nem os pools privados de memória, que dão OOM ao medir
dois braços no mesmo processo.

---

## 3. INT8: a API do PyTorch é *mais lenta* que bf16

Eu medi `torch._int_mm` na 3090, obtive 65 TOPS de 284 teóricos, e concluí "int8
não acelera aqui". **Errado.** Medi uma API, não um formato.

`torch._int_mm` obriga a quantizar fora, chamar o mm, e reescalar fora: três
kernels, dois trafegando bf16. Um kernel que quantiza a ativação **por linha
dentro dele** e aplica a escala no epílogo faz isso num lançamento só.

**[MEDIDO]**, M = 145 tokens, tempos em ms:

| GEMM | N × K | bf16 | INT8 fundido | `_int_mm` + sanduíche |
|---|---|---|---|---|
| cabeça | 155776 × 2560 | 2,302 | **0,892** (2,58x) | 3,844 (0,60x) |
| w2 (down) | 2560 × 9728 | 0,201 | **0,102** (1,98x) | 0,500 |
| w1/w3 (gate/up) | 9728 × 2560 | 0,220 | **0,119** (1,85x) | 0,339 |
| wqkv | 6144 × 2560 | 0,115 | **0,073** (1,57x) | 0,311 |
| wo | 2560 × 2560 | 0,064 | **0,055** (1,16x) | 0,196 |

O `_int_mm` fica em **0,30x–0,37x** — pior que o bf16 direto. Erro L2 do INT8:
1,3%, constante. Em 580 e 2048 tokens o ganho sobe para 2,4x–3,2x.

Isso vale para **qualquer** kernel W8A8 que quantize a ativação internamente e
receba o peso já em `int8`; o ponto não é qual biblioteca, é o formato.

### Por que o backward não precisa de `grad_weight`

Em LoRA o peso base é **congelado**. O backward só precisa de
`grad_input = grad_output @ W`. Usando o peso **bf16 original** (não a cópia
int8), esse backward é **exato** — erro 0,0 medido.

### INT8 sozinho PIORA

```
base                    0,289 s   1,00x
compile                 0,182 s   1,59x
INT8 sozinho            0,391 s   0,77x   <- PIORA
INT8 + compile          0,136 s   2,13x
```

Sem compile o passo já estava limitado por despacho (GPU ociosa 49 ms de 266).
O INT8 corta 52 ms de GEMM e a ociosidade apenas **cresce** para 103 ms. O
acelerador piorou porque acelerou: encurtou o kernel médio de 19,3 para 13,9 µs
e a fila de despacho ficou pior.

### Onde o INT8 rende, por subsistema

| | ganho sobre compile |
|---|---|
| MLP (w1/w3/w2) | **1,24x** |
| projeções de atenção (wqkv/wo) | 1,08x |
| cabeça | **1,01x — nada** |

A cabeça é o maior GEMM do modelo (2,58x isolado) e não muda o passo, porque é
**uma** chamada: economiza 1,4 ms de 182. Os 36 blocos × 5 lineares somam muito
mais. Eu tinha escrito que ela seria "o maior ganho isolado" — verdade por GEMM,
falso por passo de treino.

---

## 4. A armadilha que custou quatro baterias: `autograd.Function` sob `torch.compile`

Escrevi o backward como `torch.autograd.Function` em Python. Correto, exato, e
**anulava o ganho inteiro**:

```
braço                        GPU      parede   GPU ociosa
compile                   174,3 ms    183 ms      9 ms
compile + INT8 só cabeça  172,0 ms    247 ms     75 ms   <- DOIS lineares
compile + INT8 tudo       143,6 ms    262 ms    118 ms
```

**Dois lineares já custavam 75 ms.** Não é custo por chamada: o Dynamo não
consegue traçar o `backward()` de uma `autograd.Function` (é Python arbitrário),
embrulha em `autograd_function_apply` e marca o ponto como opaco. O AOTAutograd
então não gera o backward dentro do grafo e **o backward inteiro volta a ser
eager**.

Detalhe que explicava por que minhas medições brigavam entre si: dentro dos
blocos o estrago não aparecia, porque o `gradient_checkpointing` já torna aquele
trecho opaco. Só apareceu quando a **cabeça** entrou — ela fica fora do
checkpoint, no caminho direto do backward.

Conserto — registrar o autograd **no op**:

```python
@torch.library.custom_op("meu::linear_int8", mutates_args=())
def _linear_int8(x, qw, escala, w):
    ...  # o GEMM int8; `w` (bf16) entra só para o backward poder salvá-lo

@_linear_int8.register_fake
def _(x, qw, escala, w):
    return torch.empty(*x.shape[:-1], qw.shape[0], dtype=x.dtype, device=x.device)

def _guardar(ctx, inputs, output):      # os nomes são obrigatórios: o PyTorch
    ctx.save_for_backward(inputs[3])    # chama por palavra-chave

def _derivar(ctx, grad_out):
    (w,) = ctx.saved_tensors
    return grad_out @ w, None, None, None

torch.library.register_autograd("meu::linear_int8", _derivar,
                                setup_context=_guardar)
```

O backward vira `grad_out @ w`, um op comum que o AOTAutograd coloca no grafo.

Outra pegadinha da mesma família: chame o op **registrado**, não a função do
backend por trás dele. Só o `custom_op` com `register_fake` é traçável; chamando
a função direta, o `torch.compile` quebra o grafo em cada linear.

---

## 5. Bucket por comprimento — 1,48x, e mata o argumento do packing

O `TextDataCollator` do repo faz:

```python
max_tokens_length = min(max(len(e) for e in batch), self.max_length)
```

Preenche até o **maior exemplo do lote**, não até `max_length`. Então o
desperdício é propriedade da *variância dentro do lote*, e cresce com o batch.

**[MEDIDO]** num pack de 10.800 amostras (mediana 93, p95 184, máx 530):

```
batch    eficiencia hoje   com bucket   recupera   sobra p/ packing
    4            67,7%        99,9%      +47,5%              0,1%
    6            61,6%        99,9%      +62,3%              0,1%
    8            57,3%        99,9%      +74,4%              0,1%
```

**32% do trabalho em batch 4 é padding puro**, e ordenar por comprimento antes de
lotear elimina 99,9% dele. Em relógio:

```
batch 4, largura 137 (com padding)   0,344 s
batch 4, largura  93 (bucket)        0,233 s   1,48x
batch 8, largura 162 (com padding)   0,749 s
batch 8, largura  93 (bucket)        0,430 s   1,74x
```

**Consequência:** packing/varlen — juntar exemplos num tensor
`total_tokens × dim` com `cu_seqlens` — é a obra de engenharia mais cara que
resta, e vale **0,1%** depois do bucket. O desperdício vem da variância, e
ordenar mata a variância por construção; não sobra o que packing pegue.

Use um *bucketed shuffled sampler*, não um `sort()` global: embaralhe dentro de
cada faixa e embaralhe a ordem das faixas, senão o comprimento vira currículo.

Segundo efeito: **sem bucket, batch 8 perde para batch 4; com bucket, ganha**
(18,60 contra 17,17 amostras/s). O padding estava comendo o ganho de ocupação.

---

## 6. O que NÃO funcionou — e o padrão por trás

| tentativa | resultado |
|---|---|
| FlashAttention forçado | **1,035x.** QK/AV são 1,2% do trabalho |
| SageAttention / sparse attention | **não têm backward**; autograd não atravessa |
| Liger `fused_linear_cross_entropy` | **13x pior** em 145 tokens (custo fixo ~145 ms). Empata em 4096 com 3,7x menos VRAM |
| `sdpa_kernel(...)` + compile | **pior que `math`** — o context manager causa graph break ×36 |
| `torch._int_mm` | 0,30x, pior que bf16 |
| packing / varlen | 0,1% depois do bucket |
| CUDA Graphs | +2,2% em batch 4 |
| INT8 no backward | 1,26x, mas o cosseno do gradiente cai de 0,950 para 0,820 |

O padrão que explica quase toda a coluna da direita:

- o que otimiza a dimensão de **SEQUÊNCIA** falha com 93 tokens — flash, sparse,
  Liger, fused CE. Não existe kernel de atenção que tire 30% de um componente
  que ocupa 1,2%;
- o que otimiza o **GEMM** funciona, porque N e K continuam grandes mesmo com
  frase curta — 9728 × 2560 não encolhe quando a frase encolhe;
- o que otimiza **despacho** funciona, e é o maior de todos.

### Gradient checkpointing: reavalie depois de acelerar

Com checkpointing ligado, o forward roda duas vezes (uma real, uma recomputada
no backward). Com INT8 o passo se decompõe em forward 26 ms + recompute 26 ms +
backward 52 ms — o recompute é imposto de memória puro.

Um teste antigo dizia "desligar dá OOM (22,38 GB de 24)". **Era em largura 160,
sem bucket.** Com o bucket, largura 93:

```
checkpoint em todas as 36    0,431 s   15,64 GB   1,00x
checkpoint em 1 de cada 2    0,406 s   17,11 GB   1,06x
checkpoint em 1 de cada 3    0,396 s   18,08 GB   1,09x
desligado                    0,354 s   20,14 GB   1,22x
```

A escada é monótona — não há ponto doce intermediário, só o limite da VRAM. E o
limite é o produto **B × T**: `8 × 93 = 744` tokens cabe, `8 × 128 = 1024` dá
OOM. Isso pede política por bucket, não uma flag global.

---

## 7. Dois detalhes do S2 Pro que custam tempo de quem for mexer

**`setup_lora` liga os dois transformers.** Os alvos `attention` e `mlp`, sem
prefixo, casam tanto no Slow AR quanto no Fast AR. Só existiam os prefixos
`fast_*`. Treinar "só o Slow" não era possível sem patch — e como o Slow é quem
lê o texto, quem quer ensinar notação estava treinando a metade errada.

**`use_reentrant=True` mata o treino de LoRA em subconjunto.** A versão
reentrante do `checkpoint` só propaga gradiente se algum **input** do bloco
exigir; parâmetro interno não conta. Com LoRA apenas no Slow, a entrada vem do
embedding congelado e o treino morre com `element 0 of tensors does not require
grad`. `use_reentrant=False` resolve e é a implementação recomendada há várias
versões.

**Com `tie_word_embeddings=True`, `self.output` não existe.** O `nn.Linear` só é
criado quando a amarração está **desligada**; com ela ligada a cabeça é
`F.linear(x, embeddings.weight)` solta. Qualquer varredura por `named_modules()`
passa reto — e é o maior GEMM do modelo.

---

## 8. Como medir sem publicar ruído

Quatro erros meus, todos capazes de gerar um número que parece bom:

**Aquecimento.** A 3090 fica em 210 MHz ociosa e vai a 2100 sob carga. Com 5
passos de aquecimento o **mesmo braço** deu 0,263 / 0,263 / 0,313 s em três
processos — **19% de espalhamento**, maior que o "ganho de 1,11x" que eu ia
reportar. Use 20.

**Alternância.** Agrupar braços (A,A,B,B) faz deriva lenta virar "ganho" de quem
rodou por último. Alterne (A,B,A,B) e reporte o espalhamento *do braço de
referência* na mesma tabela. Ganho menor que o espalhamento não foi medido, foi
sorteado.

**Mínimo, não mediana, quando há interferência.** Um antivírus varrendo os
gigabytes de checkpoint que cada montagem lê sobe a 100% de um núcleo. Num passo
limitado por CPU isso entra direto no cronômetro. A interferência é aditiva e
intermitente: só pode *atrasar* um passo, nunca adiantar. Sob esse regime o
mínimo estima o tempo real e a mediana estima "tempo real + quanto a máquina
estava ocupada". Foi o mínimo que se manteve firme (0,183 / 0,184 em baterias
diferentes) enquanto a mediana da mesma medição variou de 0,184 para 0,247.

**Saída de instrumento não é prova de execução.** "201 lineares trocados" é o
meu `print`, não evidência de que o kernel rodou. A prova foi ver o kernel INT8
aparecer no profiler com 23,81 ms e o GEMM bf16 cair de 154,19 para 79,81.

---

## O que ficou em aberto

- **Nenhum treino longo rodou com INT8.** Os logits desviam 5,3% depois de 36
  camadas, a loss é idêntica, mas o cosseno do gradiente da LoRA tem mediana
  0,950 (p10 0,918). O Adam absorve erro de escala, não de direção. Antes de
  adotar, um par gêmeo com/sem e comparar CER.
- **`w1` e `w3` recebem o mesmo `x`** e cada chamada quantiza esse mesmo tensor
  de novo. Concatenar os pesos e quantizar uma vez só deveria render, e ainda
  não medi.
- O bucket acima está **medido por forma fixa no bench**, não implementado como
  sampler.
