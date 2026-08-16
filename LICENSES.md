# Licenças das fontes

O que foi efetivamente usado nos adaptadores publicados, e sob que termos.

## Modelo base

| item | licença |
|---|---|
| [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) | **Fish Audio Research License** |

Pesquisa e uso não comercial de graça, incluindo **distribuir obra derivada**.
A Seção V define explicitamente modelos de *low-rank adaptation* como Derivative
Work, então os adaptadores caem sob ela.

Quem distribui derivada deve, pela Seção IV(a):

1. entregar cópia do acordo junto;
2. incluir um `NOTICE` com a string exata
   *"This model is licensed under the Fish Audio Research License, Copyright ©
   39 AI, INC. All Rights Reserved."*;
3. exibir **"Built with Fish Audio"** de forma visível;
4. declarar o que foi alterado e como.

Uso comercial exige acordo escrito separado: business@fish.audio.

Restrição adicional da licença: não usar o modelo nem sua saída para treinar ou
melhorar modelo generativo fundacional.

## Dados de treino

| fonte | licença | uso |
|---|---|---|
| [TAGARELA](https://huggingface.co/datasets/freds0/TAGARELA) | **CC BY-NC-SA 4.0** | não comercial |
| Mozilla Common Voice PT | CC0 / permissiva | livre |

O TAGARELA veio do *Cem Mil Podcasts*. O `BY` se cumpre citando — e está citado
aqui, no card do modelo e no `NOTICE`. O `NC` se cumpre não vendendo. O `SA` é o
ponto aberto: se peso treinado conta como obra derivada do áudio é questão
juridicamente não resolvida, e a prática corrente no Hugging Face trata que não
propaga. Isto é descrição do que as licenças dizem, não parecer jurídico.

O dataset também lista, no próprio card, usos fora de escopo que valem repetir:
nada de identificação de locutor privado, verificação biométrica, vigilância,
**clonagem de voz não autorizada**, personificação ou mídia sintética enganosa.

**Nenhum áudio é redistribuído neste repositório nem no do Hugging Face.**

## Resultado

Os adaptadores são **não comerciais por dois caminhos independentes**: a Fish
Audio Research License no modelo base, e a CC BY-NC-SA 4.0 do TAGARELA nos
dados. Basta um dos dois para fechar a porta; os dois estão fechados.

## Código

O código deste repositório é MIT — ver [`LICENSE`](LICENSE), que traz também a
nota de escopo explicando o que o MIT **não** cobre.

`fixes/semantic_ref.py` é modificação do código do
[fish-speech](https://github.com/fishaudio/fish-speech) e segue a licença do
projeto original.
