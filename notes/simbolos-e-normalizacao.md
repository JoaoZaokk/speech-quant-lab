<!-- Publicado em github.com/JoaoZaokk/ptbr-audio-lab -->

# O que cada símbolo produz no s2-pro

Catálogo do que foi **medido**, não do que parece razoável. Cada linha tem a
saída literal que o modelo produziu e como ela foi obtida — ouvido em par
mínimo, ou CER sobre a bateria de 396 itens.

Regra de leitura: **empate não paga regra.** Se a forma crua sai certa, o
normalizador não encosta nela.

---

## A lei

> **Quebra o que tem símbolo órfão. Contexto salva a frase em volta, nunca o
> símbolo.**

E na forma final, depois do caso da barra:

> Não é "o símbolo é órfão". É **"o símbolo NESTE espaçamento é um token que o
> modelo nunca viu falando"**. Mesmo caractere, dois tokens, só um tem leitura.

Prova no tokenizer do s2-pro: `' /'` é o id **608** — token de prosa. `'/'`
colado é o id **14** — subword de path e de URL, que nunca apareceu em fala.
Mesmo caractere, comportamento oposto.

Corolário medido: **frase longa não salva o símbolo.** `W4A8` sai certo em
frase curta e longa; `MB/s` e `192.168.0.1` saem errados nas duas.

---

## QUEBRA — o normalizador conserta

| entrada | sai como | conserto |
|---|---|---|
| `R$ 1,00` | "Irrigura" | por extenso |
| `R$ 0,50` | "Rê Rula 50" | por extenso |
| `R$ 4.500,00` | "R4500. Deus, eu, eu, eu, eu" | por extenso |
| `R$ 250.000,00` | "20,50 mil" | por extenso |
| `192.168.0.1` | "mil novecentos e dois **pinto** cento e sessenta e oito pinto zero pinto um" | `192 ponto 168 ponto 0 ponto 1` |
| `192.168.3.1:8000` | "…três um oito mil" (porta somem) | `…, porta 8000` |
| `500MB/s` | "cincicents emebiess e **trua** emebiess" | `500 megabytes por segundo` |
| `3 m/s` | quebra | `3 metros por segundo` |
| `1.5 s` | "um ponto cinco **esse**" | `1,5 segundos` |
| `2-4 GB` | "**doisquatro** gigas" (perde a faixa) | `2 a 4 gigabytes` |
| `M == 1` | "Mei gózoum" | `M igual a 1` |
| `--dtype` | "milradd d" | `traço traço dtype` |
| `bfloat16` | "bebelobe dezesseis" | `bê float dezesseis` |
| `entrada/saída` | "entrada **be** muda tudo" (come a palavra) | `entrada barra saída` |
| `1.` em início de linha | **grunhido** — "urrrh o número…" | `Primeiro,` |
| `\n` sem pontuação antes | o modelo **deu risada** | vira `. ` |
| `•` bullet | token próprio (id 6667) | some |
| `RTX 3090` | "rtx tres meuishxnoventa" | `erre tê xis trinta e noventa` |

### A barra, que eu errei duas vezes

```
v1  colar      "entrada/saida"     -> "entrada be muda tudo"      come a palavra
v2  espaçar    "entrada / saida"   -> o modelo FALA "barra"
v3  escrever   "entrada barra saida"                              casa com a fala
```

A v2 caiu com 35 itens: **28 rejeitados, 26 transcrições trazendo a palavra
falada** — "entrada, barrada e saída", "leitura, berra e escrita", "custo,
barragem e benefício".

O áudio nunca esteve errado. **"entrada barra saída" é leitura correta de uma
barra em português.** Errada estava a referência, que não tinha a palavra.

**Lição que vale para tudo aqui: par mínimo de n=1 decide qual das duas formas
é melhor, não decide se a vencedora está certa.**

### O nome de placa, que levou seis formas

```
3090                            -> "três mil e noventa"        CERTO
RTX 3090                        -> "rtx tres meuishxnoventa"   erra os dois
R T X 3090                      -> sigla ok, número quebrado
RTX trinta e noventa            -> número ok, sigla quebrada
RTX três mil e noventa          -> sigla ok, número quebrado
erre tê xis trinta e noventa    -> 10 de 10 PERFEITAS
```

O número sozinho sai certo, então o defeito nunca foi o `3090` — é a
**vizinhança** sigla+número. E as duas formas do meio se contradizem com
entrada idêntica (`RTX` cru saiu certo numa e errado na outra), o que classifica
a região como instável, não como token quebrado. Só a união das duas metades
resolvidas estabilizou.

---

## NÃO ENCOSTA — o cru já sai certo

| entrada | sai como | por quê não mexer |
|---|---|---|
| `W4A8` | "dabliu á quatro á oito" | maiúscula+dígito é estável; separar produz "vierampereashi" |
| `INT4` `FP16` `BF16` | corretos | idem |
| `18%` `0%` `81,3%` | "dezoito por cento" | empate |
| `1920x1080` | "mil novecentos e vinte **por** mil e oitenta" | o modelo já lê `x` como "por" |
| `PyTorch` | "pái torchi" | empate — meu `PaiTorch` era gambiarra inútil |
| `SSD` `GPU` `CPU` `API` | corretos | empate, cru e soletrado |
| `3090` sozinho | "três mil e noventa" | número nu é estável |
| `01/01/2026` | correto | normalizar data PIOROU: CER 0,041 → 0,181 |
| número nu | correto | normalizar PIOROU: 0,157 → 0,261 |
| `24/7` `C:/x` `http://` | intactos | as guardas da barra excluem |

**Três regras foram para o lixo assim que alguém escutou:** separar sigla
(0,083 → 0,292), normalizar data, normalizar número.

---

## AINDA QUEBRA E NÃO TEM CONSERTO CONHECIDO

Medido na bateria de 396, voz William Bonner. Não há regra para estes:

| entrada | sai como |
|---|---|
| `config.json` | "com o feed **pontubation**" |
| `arquivo.txt` | "arquivo o ponto **otisqui**" |
| `weights.gguf` | "8 Stuntbank F" |
| `../output/final` | "Aide deu a input final" |
| `ID 123456789` | "de **modis** 3, 4, 5, 6, 7, 8, 9" |
| `commit c4d3f3e1f` | "comit 4DF13M FIA" |
| `chamado 001927` | "001 dessa ano do start" |
| `TensorRT` | "intenso o herd" |
| `PostgreSQL` | "pós-cresgresse" |
| `torch.compile` | "torte com o pickable" |
| `R$ 1.287,43` (valor grande) | "R1200 e CTST Z243" |

Por isso o **v8 corta ruído de dígito na fonte**: sem QC, isso entraria no
dataset calado.

---

## Não-determinismo

O defeito **não é estável**. `192.168.0.1` foi gerado 4 vezes e deu 4 saídas
erradas diferentes. `RTX` cru saiu certo e errado com o mesmo texto. Uma lista
numerada melhorou ao repetir ("vai aquecendo").

Consequência prática: **regerar até passar não vale.** Em produção a primeira
saída é a que conta.

---

## Cinco idiomas apareceram no lugar de um símbolo

Quando o símbolo não tem leitura em português, o modelo o pronuncia **como
palavra**, na língua onde o embedding cair:

```
"1"   -> "Ichi-Pu"        japonês
"11"  -> "XU YI QI"       chinês
"7"   -> "Sa-chi-gu-o-su-nu-ra-ni-yo-go"
"\n"  -> "blinhinhú"      russo
"."   -> "punto"/"pinto"  espanhol/italiano
"1"   -> "wan"            inglês
"3"   -> "trua"/"túa"     francês (trois)
```

---

## Tags de colchete — não têm regra

Investigado em 2026-08-20 contra o **s2.1-pro da API** (não o s2-pro local).
Quatro hipóteses foram levantadas e as quatro morreram, cada uma no teste
seguinte.

> **Correção de 2026-08-21 — a conclusão original era larga demais.**
> Eu havia escrito "não existe regra de gatilho". O model card oficial do
> s2-pro diz:
>
> > "Rather than relying on a fixed set of predefined tags, S2 Pro accepts
> > **free-form textual descriptions** — such as `[whisper in small voice]`,
> > `[professional broadcast tone]`, or `[pitch up]`"
> > "15,000+ unique tags supported"
>
> As ~30 gerações usaram **só palavra solta** (`[whisper]`, `[laughing]`,
> `[sad]`) — que o card também lista, então não era forma inválida. Mas a
> **descrição livre nunca foi testada**, e é plausível que uma frase condicione
> mais forte por carregar mais sinal semântico.
>
> O que os testes sustentam: **tag de palavra solta é não-determinística no
> s2.1-pro da API.** O que eles NÃO sustentam: que não exista forma de
> controle. Hipótese aberta, não fechada.

| hipótese | como morreu |
|---|---|
| precisa de fronteira de segmento | tag no início do texto e depois de `.` deram zero |
| a pontuação dispara (`!`) | `inteira! [whisper] …` deu zero |
| o texto tem que concordar com a tag | `[clear throat]` disparou em "caiu na gargalhada" |
| tem que estar colada (`]E`, não `] E`) | contagem 5×: colado 0/5 em whisper, espaçado 2/5 |

A última só morreu **contando**. Com n=1 a colagem parecia 3 de 3.

### O que realmente acontece

Mesma entrada, cinco rodadas, `[sad][whispering]Eu tenho tanta saudade de
você...`:

```
1  melancólico
2  tentou sussurrar, terminou falando alto
3  só o sussurro, sem tristeza
4  "virou um intérprete de Shakespeare"
5  nada
```

Cinco interpretações **qualitativamente diferentes**. Não é gatilho que dispara
ou não — é uma **distribuição de performances**, e a tag desloca a distribuição
em vez de escolher um ponto dela.

Amarra com o resto: os 44% de alongamento que uma tag INVENTADA produz
(deslocamento mecânico, sem semântica nenhuma) são a mesma coisa vista pelo
lado da duração. Ver [[colchete-sozinho-ja-alonga]].

    tag combinada funciona          `[sad][whispering]` empilhado rende
    tag de som > tag de emoção      `[clear throat]` e `[laughing]` dá para
                                    auditar de ouvido; `[embarrassed]` não —
                                    ninguém sabe dizer como deveria soar

### Consequência para dataset

**Tag não entra em rodada sem QC.** Com "algo aconteceu" em 4/5 e "aconteceu a
coisa certa" em 0 a 2/5, metade de uma família de tags entraria pareada com um
texto que promete emoção que o áudio não tem — e sem gate ninguém saberia
quais. É conhecimento de INFERÊNCIA (gere, ouça, regere), não de corpus.

### Armadilha de medição que apareceu aqui

O alvo foi pontuado errado por várias rodadas porque `whispering` estava sendo
confundido com choro. `whispering` é falar **sem vibrar as cordas vocais** —
sopro, volume baixo. Antes de contar acerto de tag, definir o alvo audível;
senão a contagem mede outra coisa. Mesmo problema de
[[metricas-que-mediram-a-coisa-errada]].

---

## Duas coisas sobre medir, que custaram caro

**1. "Passou" não é transcrição.** Mesmo item, duas rodadas:

```
"ouve e diz se passou"   -> "1 - ok"
"transcreve LITERAL"     -> "urrrh (o um mais esquisito que já vi) o número…"
```

O grunhido virou "ok" na primeira passada porque a frase era inteligível — o
defeito estava no marcador, que o ouvido descarta como ruído de gravação. Essa
segunda passada salvou 99 itens de serem cortados por engano.

**Pedir transcrição literal, nunca nota de aprovação.**

**2. A trava de duração tinha a forma errada, não só a constante.** Ela modela
`esperado = chars / cps`, reta pela origem. O MP3 da API traz silêncio fixo nas
pontas. Ajuste sobre 34 MP3 reais:

```
duração = chars / 23,4 + 1,79 s
```

A taxa aparente **sobe** com o tamanho — 7,9 c/s aos 21 chars, 17,5 aos 58. Com
reta pela origem os dois extremos encostam na trava por causa da **forma** do
modelo, não por defeito do áudio.

> **Correção de 2026-08-21.** Eu escrevia aqui que o `+1,79 s` era "assinatura
> de padding constante". **Medido em 4 vozes, 260 MP3 da API: o silêncio nas
> pontas é 0,051 s no início e 0,03 s no fim — 1,4% da duração.** Os 0,051 s
> são idênticos em todas as vozes, o que os identifica como atraso do encoder
> MP3, não como silêncio de fala. Não existe padding de 1,8 s.
>
> O ajuste afim continua válido, foi medido. **A causa que eu atribuí a ele
> não sobreviveu, e segue sem explicação.**

É por isso que o v8 usa esse modelo como filtro de entrada, e não o `--speech-cps`.

---

As regras acima foram implementadas como uma cascata de normalização
PT-BR que roda **antes** do texto chegar ao modelo. O código dessa cascata não
está publicado — depende do pipeline de dados local e não ajudaria de fora. O
que serve a terceiros é a tabela: **qual entrada quebra, o que ela produz, e
qual forma escrita o modelo lê certo.**

Vale para qualquer front-end que alimente o s2-pro, em qualquer linguagem.
