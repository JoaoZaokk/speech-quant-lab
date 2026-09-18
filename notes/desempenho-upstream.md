<!-- Publicado em github.com/JoaoZaokk/speech-quant-lab -->

# Três coisas de terceiros que aceleram o S2 Pro e ninguém mergeou

Levantado em 2026-08-21 varrendo issues e PRs abertos do
[`fishaudio/fish-speech`](https://github.com/fishaudio/fish-speech).

**Nada aqui é trabalho meu.** É um mapa: os três achados estão em PRs abertos
há semanas, o crédito é dos autores, e cada um tem número para conferir na
fonte. Registro porque `git log` não os mostra — nenhum virou commit, porque
nenhum foi aceito.

> Se você olhou o histórico de commits do upstream e concluiu "está tudo em
> dia", olhou o que os mantenedores aceitaram, não o que a comunidade resolveu.
> Foi exatamente o erro que me fez perder isso na primeira passada.

---

## 1. `max_seq_len` — 5,6 GB de VRAM por um número no `config.json`

**Fonte:** [PR #1306](https://github.com/fishaudio/fish-speech/pull/1306), por
`moduvoice`. Notas verificadas em Tesla T4 (16 GB).

`setup_caches(max_seq_len=model.config.max_seq_len)` é chamado
**incondicionalmente** logo depois de carregar o modelo, sem olhar o
comprimento real da entrada. O cache é alocado para 32768 tokens mesmo quando o
prompt tem 234.

    max_seq_len = 32768 (padrão)   pico 15.593 MiB   OOM em 16 GB
    max_seq_len =  4096 (editado)  pico  9.939 MiB   roda, ~6,0 tok/s

O conserto é editar `text_config.max_seq_len` no `config.json` do checkpoint.
**Zero código.**

Nota do autor que economiza tempo: `--half` roda mas não ganha nada — fp16 e
bf16 gastam os mesmos 2 bytes por parâmetro.

Contexto: a documentação oficial recomenda "pelo menos 24GB para inferência".
Esses 5,6 GB mudam quais placas conseguem rodar.

## 2. Fatiar o KV cache — 1,97x **em cima** do `torch.compile`

**Fonte:** [issue #1310](https://github.com/fishaudio/fish-speech/issues/1310) e
[PR #1312](https://github.com/fishaudio/fish-speech/pull/1312), por
`quantumxiaol`.

Mesma raiz do item 1, atacada por código em vez de configuração: cada passo de
decode atende a **capacidade física** do cache, não o prefixo preenchido. Com
prompt de 234 tokens e ~400 de contexto, 0,7–1,3% do cache está ativo.

O núcleo do patch, em `Attention.forward`:

```python
if self.kv_cache is not None:
    k, v = self.kv_cache.update(input_pos, k, v)
    if mask is not None:
        # Keep the cache allocated at max_seq_len, but repeat and
        # attend only to the prefix populated so far.
        active_kv_len = mask.shape[-1]
        k = k[:, :, :active_kv_len]
        v = v[:, :, :active_kv_len]
```

E a máscara passa a ser cortada no comprimento ativo em `forward_generate`, com
`kv_len` descendo desde `decode_n_tokens` — que passa a posição em Python
justamente para **não** chamar `.item()` no caminho quente, que sincronizaria o
device.

Números do autor (Quadro GV100, FP16, s2-pro):

    modo             antes         depois       ganho
    eager          2,73 tok/s   12,57 tok/s     4,60x
    torch.compile 16,65 tok/s   32,74 tok/s     1,97x
    pico VRAM       17,33 GB      15,16 GB

Apple M4 Pro: 4,08 → 8,31 frames/s (2,04x).

**Os itens 1 e 2 são complementares, não alternativos** — um corta a alocação,
o outro corta o trabalho por passo.

## 3. Windows + `torchaudio >= 2.9` — armadilha adiada

**Fonte:** [PR #1316](https://github.com/fishaudio/fish-speech/pull/1316), por
`jamesonBradfield`. Este foi fechado, não mergeado.

`torchaudio >= 2.9` **ignora o argumento `backend=`** e sempre decodifica via
TorchCodec, que exige FFmpeg em shared library — coisa que costuma faltar no
Windows. O conserto troca por `soundfile`, que traz o libsndfile embutido, em
`fish_speech/inference_engine/reference_loader.py`.

Não morde quem está em `torchaudio 2.8`. Morde no dia da atualização.

---

## Sobre `torch.compile` no Windows

Nota separada, porque é erro meu documentado: **`triton-windows` existe e
funciona.** `pip install triton-windows==3.4.0.post21` casa com torch 2.8, e
`torch.utils._triton.has_triton()` passa a devolver `True`.

Medido no s2-pro, decode autorregressivo:

    sem compile   29,3 s e 32,8 s
    com compile   10,0 s e 11,4 s      2,9x

Transcrição idêntica palavra por palavra, CER 0,000 nos dois caminhos. Não é
aproximação — é o mesmo áudio mais rápido. A primeira geração paga ~318 s de
compilação e o ganho começa na segunda; paga a si mesma por volta da 16ª.

Custo: a VRAM sobe de ~10 GB para 16,7 GB. **É aí que os itens 1 e 2 acima
deixam de ser luxo.**

Eu havia escrito num comentário de código que "Triton não roda no Windows", a
partir de um aviso de log, e nunca conferi. Isso fez toda geração do projeto
custar 3x do necessário.
