"""Troca os `nn.Linear` CONGELADOS do Fish por GEMM INT8 do comfy-kitchen.

O QUE ISTO E, E O QUE NAO E
---------------------------
Nao e' quantizacao do modelo. O checkpoint continua bf16 e o merge continua
saindo bf16. Isto e' aceleracao do PASSO DE TREINO: o peso base esta' congelado,
entao a copia int8 dele e' constante durante o treino inteiro e pode ser
calculada uma vez.

POR QUE FUNCIONA AQUI E NAO FUNCIONOU COM `torch._int_mm`
--------------------------------------------------------
MEDIDO em 2026-09-18 (`scripts/bench_int8_kernel.py`, RTX 3090 sm86), M=145:

    GEMM              bf16    int8 ck   _int_mm+sanduiche
    head 155776x2560  2,302     0,892   3,844      <- 2,58x contra 0,60x
    w1/w3  9728x2560  0,220     0,119   0,339      <- 1,85x contra 0,65x
    w2     2560x9728  0,201     0,102   0,500

`torch._int_mm` exige quantizar fora, chamar o mm, e reescalar fora: tres
kernels, dois deles trafegando bf16. Fica MAIS LENTO que o bf16 direto. O
`comfy_kitchen::int8_linear` quantiza a ativacao por linha DENTRO do kernel
cuBLASLt e aplica a escala no epilogo — um lancamento so'.

O QUE O KERNEL NAO TEM
----------------------
Backward. E' um `torch.library.custom_op` sem regra de autograd, entao
`out.requires_grad` sai False. Aqui isso nao impede nada, porque o peso base
NAO TREINA: so' e' preciso `grad_input = grad_output @ W`, nunca `grad_weight`.
O op `fish_accel::linear_int8` abaixo fornece exatamente isso.

SOZINHO ELE PIORA — E ISSO NAO E' DEFEITO
-----------------------------------------
MEDIDO no passo de treino (160 tokens, batch 1, LoRA slow, bracos ALTERNADOS,
ganho sobre o MINIMO, espalhamento 0,1% a 0,7%):

    base                    0,289 s   1,00x
    compile                 0,182 s   1,59x
    compile + int8 MLP      0,147 s   1,97x
    compile + int8 atencao  0,169 s   1,71x
    compile + int8 cabeca   0,181 s   1,60x   (nao mexe)
    compile + int8 tudo     0,136 s   2,13x   <- melhor
    int8 SOZINHO, sem compile         0,77x   <- PIORA

O perfil (`scripts/perfil_passo_fish.py`) diz por que o INT8 sozinho piora: sem
compile o passo gasta 217 ms de GPU num passo de 266 ms de PAREDE — a GPU ja'
esperava a CPU. O INT8 corta 52 ms de GEMM e a ociosidade apenas cresce (49 ->
100 ms). Com `compile` a GPU passa a ficar saturada (174 de 183 ms) e ai' o
tempo de GPU economizado vira tempo de relogio.

Consequencia pratica: **este modulo nao deve ser ligado sem `--compile`.**

Onde o ganho esta', e nao esta':
- MLP (w1/w3/w2) e' 73% do trabalho e rende 1,24x sobre o compile;
- as projecoes de atencao rendem 1,08x;
- a CABECA nao rende nada (1,01x). E' o maior GEMM do modelo em uma chamada,
  mas e' UMA chamada: economiza 1,4 ms de 182. Os 36 blocos x 5 lineares somam
  muito mais. Eu tinha escrito que ela seria "o maior ganho isolado" — era
  verdade por GEMM, falso por passo de treino.

O `use_gradient_checkpointing` esta' LIGADO e REFAZ o forward de cada bloco no
backward, entao cada camada roda o forward duas vezes por passo — e' por isso
que acelerar so' o forward ja' paga.

O QUE FICA DE FORA, DE PROPOSITO
--------------------------------
- A LoRA (`lora_A`/`lora_B`): r=32, custo irrelevante, e e' ela que treina.
  `loralib.Linear.forward` soma os dois caminhos; so' o base vira int8.
- O BACKWARD, por padrao. Com gradient checkpointing o forward roda 2x e o
  backward 1x, entao dos 155,8 ms de GEMM do passo ~104 sao forward (esses
  aceleram) e ~52 sao backward (esses nao). Existe `backward_int8=True`, que
  funciona e e' 1,26x mais rapido, mas MEDIDO em 2026-09-18 ele NAO COMPENSA:

        backward   cosseno mediana   p10     s/passo   VRAM
        bf16             0,950      0,918     0,135   15,64 GB
        int8             0,820      0,798     0,107   17,59 GB

  Em angulo, o gradiente sai de 18 para 35 graus fora da direcao certa, e o
  erro relativo dobra (0,32 -> 0,64). Num projeto que ja' apanhou de
  esquecimento catastrofico, 26% de velocidade nao paga gradiente torto que
  so' aparece no eval horas depois. Fica implementado e DESLIGADO.

O QUE ISSO CUSTA EM PRECISAO
----------------------------
MEDIDO (`scripts/erro_int8_fish.py`, `int8_tudo`, 160 tokens, primeiro passo):

    GEMM isolado, erro L2                    1,3%
    logits, depois de 36 camadas             5,3%
    loss                          identica ate' a 4a casa
    gradiente da LoRA: cosseno     mediana 0,950   p10 0,918   pior 0,318

(medido com a versao `autograd.Function`; o op atual tem a mesma matematica no
forward e o MESMO backward exato, entao os numeros valem — mas vale remedir se
alguem mexer no `_derivar`.)

O backward e' EXATO (erro 0,0 no teste unitario) porque usa o peso bf16
original; todo o desvio vem do forward, propagado camada a camada. Os piores
cossenos estao nas camadas 0 a 2 — as que recebem o gradiente depois de ele
atravessar as 36.

Cosseno importa mais que magnitude: o Adam normaliza pelo segundo momento e
absorve erro de escala, mas nao corrige erro de DIRECAO. 0,95 e' utilizavel;
nao e' de graca. Nenhum treino longo foi feito com isto ainda.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ERRO_L2_MEDIDO = 0.0128  # per-channel, formas do s2-pro, 2026-09-18

# `torch.ops.comfy_kitchen.int8_linear`, e nao `backends.cuda.int8_linear`.
# Os dois chamam o mesmo kernel, mas so' o primeiro e' um `torch.library.custom_op`
# com `register_fake`, entao so' ele o Dynamo consegue tracar. Chamando a funcao
# do backend direto, `torch.compile` quebra o grafo em cada linear — 201 quebras,
# e o compile e' justamente o que precisa funcionar aqui (MEDIDO: o passo esta'
# limitado por lancamento de kernel, com a GPU ociosa 49 ms de 266).
_CODIGO_DTYPE: dict[torch.dtype, int] = {}


def _codigo(dtype: torch.dtype) -> int:
    if not _CODIGO_DTYPE:
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

        _CODIGO_DTYPE.update(DTYPE_TO_CODE)
    return _CODIGO_DTYPE[dtype]


# POR QUE UM OP PROPRIO, E NAO UMA `torch.autograd.Function`
# ----------------------------------------------------------
# A primeira versao usava `autograd.Function`. Funcionava e era exata, mas
# MEDIDO em 2026-09-18 ela custava caro sob `torch.compile`:
#
#     braco                  GPU       parede    GPU ociosa
#     compile             174,3 ms     183 ms       9 ms   <- saturada
#     int8_cabeca (2!)    172,0 ms     247 ms      75 ms
#     int8_tudo (202)     143,6 ms     262 ms     118 ms
#
# O INT8 economizava GPU de verdade (174 -> 143 ms) e ainda assim o relogio
# PIORAVA. E com DOIS lineares o estrago ja' era de 75 ms — ou seja, nao e'
# custo por chamada: o Dynamo trata `autograd.Function` customizada como
# `autograd_function_apply`, o AOTAutograd nao consegue gerar o backward dentro
# do grafo, e o backward INTEIRO volta a ser eager.
#
# Registrando o autograd NO OP (`torch.library.register_autograd`), o backward
# vira um op comum: o AOTAutograd o coloca no grafo e o compile o otimiza junto
# com o resto.
_ASSINATURA = "fish_accel::linear_int8"


@torch.library.custom_op(_ASSINATURA, mutates_args=())
def _linear_int8(x: torch.Tensor, qw: torch.Tensor, escala: torch.Tensor,
                 w: torch.Tensor) -> torch.Tensor:
    # `w` (bf16) nao e' usado no forward: entra so' para o backward poder
    # salva-lo. Um op nao pode fechar sobre tensores, tudo passa por argumento.
    return torch.ops.comfy_kitchen.int8_linear(
        x, qw, escala, None, _codigo(x.dtype), False, 256, None)


@_linear_int8.register_fake
def _(x, qw, escala, w):
    return torch.empty(*x.shape[:-1], qw.shape[0], dtype=x.dtype, device=x.device)


def _guardar(ctx, inputs, output):
    # Os nomes `ctx`, `inputs` e `output` sao obrigatorios: o PyTorch chama este
    # callback por palavra-chave (`torch/_library/autograd.py`), entao traduzi-los
    # para portugues derruba com `unexpected keyword argument`.
    ctx.save_for_backward(inputs[3])            # o peso bf16


def _derivar(ctx, grad_out):
    """`grad_input = grad_output @ W`, em bf16 sobre o peso original.

    Nao ha' `grad_weight`: o peso base do Fish e' congelado. Usar o peso bf16
    (e nao a copia int8) torna este backward EXATO — MEDIDO: erro 0,0.
    """
    (w,) = ctx.saved_tensors
    return grad_out @ w, None, None, None


torch.library.register_autograd(_ASSINATURA, _derivar, setup_context=_guardar)


# ---------------------------------------------------------------------------
# Variante com o BACKWARD tambem em INT8.
#
# Op separado, e nao um argumento: `register_autograd` registra UMA regra por
# op. Um `if` dentro do backward nao existiria — a regra ja' esta' escolhida
# quando ele roda.
#
# MEDIDO: com gradient checkpointing o forward roda 2x e o backward 1x, entao
# dos 155,8 ms de GEMM do passo cerca de 104 ms sao forward e 52 ms backward.
# A variante de cima acelera so' os 104; esta ataca os 52 restantes.
#
# O PRECO: aqui o gradiente passa a ser quantizado, e gradiente quantizado e'
# onde a precisao costuma morrer. A variante de cima e' EXATA (erro 0,0); esta
# nao e'. Medir antes de usar.
_ASSINATURA_B8 = "fish_accel::linear_int8_b8"


@torch.library.custom_op(_ASSINATURA_B8, mutates_args=())
def _linear_int8_b8(x: torch.Tensor, qw: torch.Tensor, escala: torch.Tensor,
                    qw_t: torch.Tensor) -> torch.Tensor:
    return torch.ops.comfy_kitchen.int8_linear(
        x, qw, escala, None, _codigo(x.dtype), False, 256, None)


@_linear_int8_b8.register_fake
def _(x, qw, escala, qw_t):
    return torch.empty(*x.shape[:-1], qw.shape[0], dtype=x.dtype, device=x.device)


def _guardar_b8(ctx, inputs, output):
    ctx.save_for_backward(inputs[2], inputs[3])     # escala e o peso transposto


def _derivar_b8(ctx, grad_out):
    """`grad_input = grad_output @ W`, com o GEMM tambem em INT8.

        grad_x[m,k] = soma_n grad_out[m,n] * qw[n,k] * escala[n]

    A escala corre sobre `n`, que e' o indice SOMADO — entao ela nao cabe no
    epilogo do kernel, que so' escala a saida. Tem de entrar no `grad_out`
    ANTES. Depois o GEMM roda com o peso transposto e escala unitaria, e o
    proprio kernel quantiza `g` por linha.
    """
    escala, qw_t = ctx.saved_tensors
    g = (grad_out * escala.to(grad_out.dtype)).contiguous()
    um = torch.ones(qw_t.shape[0], dtype=torch.float32, device=g.device)
    grad_x = torch.ops.comfy_kitchen.int8_linear(
        g, qw_t, um, None, _codigo(g.dtype), False, 256, None)
    return grad_x, None, None, None


torch.library.register_autograd(_ASSINATURA_B8, _derivar_b8,
                                setup_context=_guardar_b8)


def _aplicar_gemm(x, qw, escala, w, qw_t):
    """Escolhe a variante. `qw_t` presente significa backward em INT8."""
    if qw_t is not None:
        return torch.ops.fish_accel.linear_int8_b8(x, qw, escala, qw_t)
    return torch.ops.fish_accel.linear_int8(x, qw, escala, w)


class LinearINT8(nn.Module):
    """Substitui um `nn.Linear` congelado, mantendo `weight` no `state_dict`.

    O `weight` original continua sendo um Parameter com o mesmo nome e a mesma
    forma. Isso nao e' detalhe: `merge_fish_lora.py`, o `setup_lora` e o
    `on_save_checkpoint` do `lit_module` procuram pesos POR NOME. Trocar o
    modulo sem preservar o nome quebraria os tres em silencio.
    """

    def __init__(self, base: nn.Linear, per_channel: bool = True,
                 backward_int8: bool = False):
        super().__init__()
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as INT8

        if base.bias is not None:
            raise ValueError("o Fish nao usa bias nos lineares; recebi um com bias")

        self.weight = base.weight              # mesmo Parameter, mesmo nome
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.backward_int8 = backward_int8

        qw, p = INT8.quantize(base.weight.detach(), is_weight=True,
                              per_channel=per_channel)
        escala = p.scale.to(torch.float32).reshape(-1).contiguous()

        # Buffers, nao Parameters: nao recebem gradiente nem entram no
        # otimizador. `persistent=False` os mantem FORA do state_dict, senao o
        # checkpoint ganharia uma copia int8 do modelo inteiro.
        self.register_buffer("_qw", qw, persistent=False)
        self.register_buffer("_escala", escala, persistent=False)
        self._qw_t = qw.t().contiguous() if backward_int8 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _aplicar_gemm(x, self._qw, self._escala, self.weight, self._qw_t)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, int8, "
                f"backward={'int8' if self.backward_int8 else 'bf16'}")


def _base_int8(self, x: torch.Tensor) -> torch.Tensor:
    """`forward` novo da `loralib.Linear`: base em int8, LoRA intacta em bf16."""
    saida = _aplicar_gemm(x, self._qw, self._escala, self.weight, self._qw_t)
    if self.r > 0 and not self.merged:
        saida = saida + (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                         @ self.lora_B.transpose(0, 1)) * self.scaling
    return saida


CABECAS = ("output", "fast_output")


def aplicar(modelo: nn.Module,
            alvos: tuple[str, ...] = ("w1", "w3", "w2", "wqkv", "wo"),
            cabeca: bool = True, per_channel: bool = True,
            backward_int8: bool = False, verbose: bool = True) -> dict:
    """Troca os lineares indicados por INT8, no lugar.

    `alvos` sao os SUFIXOS do nome qualificado, nao substrings: "w1" casa
    `layers.0.feed_forward.w1` mas nao casaria um `w1x` hipotetico. O casamento
    por sufixo pega tambem o Fast AR (`fast_layers.N.feed_forward.w1`), que e'
    o desejado — sao os mesmos GEMMs, so' que em 4 camadas em vez de 36.

    `cabeca` cobre a `output` (2560 x 155.776), que MEDIDO custa 10,859 ms —
    3,7x um bloco transformer inteiro — e e' o maior ganho isolado da tabela;
    e a `fast_output` dos codebooks, bem menor.
    """
    import loralib
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as INT8

    trocados, ignorados = [], []

    for nome, mod in list(modelo.named_modules()):
        sufixo = nome.rsplit(".", 1)[-1]
        if not (sufixo in alvos or (cabeca and sufixo in CABECAS)):
            continue

        if isinstance(mod, loralib.Linear):
            # Modulo LoRA: nao se troca o modulo — o `setup_lora` e o
            # `mark_only_lora_as_trainable` ja' o registraram, e trocar o objeto
            # desfaria os dois. Anexa-se a copia int8 do base e troca-se so' o
            # metodo `forward`.
            if mod.bias is not None:
                ignorados.append((nome, "tem bias"))
                continue
            qw, p = INT8.quantize(mod.weight.detach(), is_weight=True,
                                  per_channel=per_channel)
            mod.register_buffer("_qw", qw, persistent=False)
            mod.register_buffer("_escala",
                                p.scale.to(torch.float32).reshape(-1).contiguous(),
                                persistent=False)
            mod.register_buffer("_qw_t",
                                qw.t().contiguous() if backward_int8 else None,
                                persistent=False)
            mod._backward_int8 = backward_int8
            mod.forward = _base_int8.__get__(mod, type(mod))
            trocados.append((nome, "lora+int8"))

        elif isinstance(mod, nn.Linear):
            pai = modelo.get_submodule(nome.rsplit(".", 1)[0]) if "." in nome else modelo
            setattr(pai, sufixo, LinearINT8(mod, per_channel, backward_int8))
            trocados.append((nome, "int8"))

    if cabeca:
        trocados += _cabeca_amarrada(modelo, per_channel, backward_int8)

    if verbose:
        print(f"int8: {len(trocados)} lineares trocados"
              + (f", {len(ignorados)} ignorados" if ignorados else ""))
        for nome, motivo in ignorados:
            print(f"  ignorado {nome}: {motivo}")

    return {"trocados": trocados, "ignorados": ignorados,
            "erro_l2_esperado": _ERRO_L2_MEDIDO}


def _cabeca_amarrada(modelo: nn.Module, per_channel: bool,
                     backward_int8: bool) -> list[tuple[str, str]]:
    """Acelera a cabeca de token quando ela e' `embeddings.weight` amarrado.

    Com `tie_word_embeddings=True` — o caso do s2-pro — nao existe
    `self.output`: o `if` do vendor so' cria o `nn.Linear` quando a amarracao
    esta' DESLIGADA. A cabeca e' `F.linear(x, embeddings.weight)`, que o
    `named_modules()` nao ve' por nao ser modulo. E' o MAIOR GEMM do modelo.

    O mesmo tensor serve de tabela de embedding (consulta por indice) e de
    cabeca (GEMM). So' o GEMM passa a int8; a consulta continua em bf16, porque
    quantizar a tabela mudaria a ENTRADA do modelo, nao so' a saida.
    """
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as INT8

    alvo = modelo
    if not (getattr(alvo, "config", None)
            and getattr(alvo.config, "tie_word_embeddings", False)
            and hasattr(alvo, "embeddings")
            and hasattr(alvo, "cabeca_token")):
        return []

    w = alvo.embeddings.weight
    qw, p = INT8.quantize(w.detach(), is_weight=True, per_channel=per_channel)

    # Buffers no modulo, nao variaveis presas na closure: assim um `.to(device)`
    # ou `.half()` posterior os move junto. `persistent=False` os mantem fora do
    # state_dict — o checkpoint nao deve carregar copia int8 de nada.
    alvo.register_buffer("_qw_cabeca", qw, persistent=False)
    alvo.register_buffer("_escala_cabeca",
                         p.scale.to(torch.float32).reshape(-1).contiguous(),
                         persistent=False)
    alvo.register_buffer("_qw_t_cabeca",
                         qw.t().contiguous() if backward_int8 else None,
                         persistent=False)

    def cabeca_int8(x: torch.Tensor) -> torch.Tensor:
        return _aplicar_gemm(x, alvo._qw_cabeca, alvo._escala_cabeca,
                             alvo.embeddings.weight, alvo._qw_t_cabeca)

    alvo.cabeca_token = cabeca_int8
    return [("cabeca_token (embeddings amarrado)", f"int8 {tuple(w.shape)}")]
