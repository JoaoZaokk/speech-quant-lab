"""Varre uma arvore antes de publicar, procurando o que nao pode sair da maquina.

POR QUE ISTO EXISTE
-------------------
Uma bancada e um repositorio publico vivem no mesmo diretorio, e o que separa um
do outro nao e' o bom senso na hora do `git add` — e' uma lista. Numa publicacao
anterior deste projeto subiram 108 arquivos de uma vez, incluindo dois documentos
internos que carregavam o mapa de discos da maquina. Foi preciso apagar o
repositorio e o `.git` local: historico guarda tudo, reaproveitar nao adianta.

Isto NAO substitui ler o que se publica. Pega o vazamento MECANICO — caminho de
disco, nome de usuario, caminho de dado — que passa despercebido justamente por
ser detalhe de ambiente que o autor ja' nem enxerga.

O FALSO POSITIVO E' O INIMIGO
-----------------------------
Um padrao que dispara em `https://` faz o relatorio virar ruido, e relatorio
ruidoso nao e' lido — o que e' pior do que nao ter relatorio, porque da' a
sensacao de ter conferido. Dai' duas decisoes de projeto:

- `(?<![A-Za-z0-9])[A-Za-z]:[/\\]` para caminho de disco: uma letra de drive tem
  UMA letra antes dos dois pontos, nao "http". Sem esse olhar-para-tras o
  scanner acusa toda URL do documento.
- toda isencao mora numa lista com o motivo escrito ao lado. Isencao sem
  justificativa vira porta dos fundos permanente.

    python bench/varrer_antes_de_publicar.py caminho/do/repo
    python bench/varrer_antes_de_publicar.py . --segredo meu_projeto --segredo pack_interno

Sai com codigo 1 se achou algo, para poder entrar num hook de pre-commit.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# Padroes que valem para qualquer projeto. Nomes especificos do seu (pastas de
# dado, nomes de dataset) entram por `--segredo`.
PADROES: dict[str, re.Pattern[str]] = {
    # O `(?<![A-Za-z0-9])` e' o que separa `C:/Users` de `https://`.
    "caminho de disco": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\]"),
    "compartilhamento de rede": re.compile(r"(?<![:\w])(//|\\\\)[A-Za-z][\w.-]{2,}[/\\]"),
    "home de usuario": re.compile(r"AppData|Users[/\\]|/home/\w|/Users/\w", re.I),
    "e-mail": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    "chave ou token": re.compile(
        r"(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
        r"|AKIA[A-Z0-9]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY)"),
    # Caminho PARA dado de saida, nao a extensao solta nem pasta de codigo:
    # uma pasta de saida com arquivo dentro vaza; `*.ckpt` num `.gitignore` e'
    # o contrario de vazar, e `fish_speech/datasets/` e' codigo de terceiro.
    "caminho de dado": re.compile(
        r"(exporters|runs|checkpoints|data/packs)[/\\]\w", re.I),
}

# Arquivos cuja razao de existir e' listar o que fica de fora.
ISENTOS_POR_ARQUIVO = (".gitignore", ".gitattributes", "LICENSE", "LICENSES.md")

# Este arquivo contem os proprios padroes que procura, entao casa consigo mesmo.
# Isenta-lo inteiro seria uma porta dos fundos num arquivo editavel — bastaria
# alguem colar um caminho num comentario daqui para ele sumir do relatorio para
# sempre. As duas linhas que casam vao para `.publicacao-isencoes`, onde ficam
# visiveis e revisaveis como qualquer outra isencao.


def carregar_isencoes(caminho: pathlib.Path) -> list[str]:
    """Trechos liberados, um por linha, `#` comenta.

    Cada linha deveria vir com o motivo ao lado; o scanner nao cobra, mas quem
    ler o arquivo daqui a seis meses vai precisar.
    """
    if not caminho.exists():
        return []
    linhas = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        trecho = linha.split("#", 1)[0].strip()
        if trecho:
            linhas.append(trecho)
    return linhas


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raiz", nargs="?", default=".")
    ap.add_argument("--segredo", action="append", default=[],
                    help="termo extra a procurar (nome de dataset, de pasta "
                         "interna, do projeto). Pode repetir.")
    ap.add_argument("--isencoes", default=".publicacao-isencoes",
                    help="arquivo com trechos liberados, um por linha")
    ap.add_argument("--quieto", action="store_true",
                    help="so' o resumo; util em hook de pre-commit")
    cli = ap.parse_args()

    raiz = pathlib.Path(cli.raiz)
    if not raiz.is_dir():
        print(f"{raiz} nao e' um diretorio")
        return 2

    padroes = dict(PADROES)
    for termo in cli.segredo:
        padroes[f"termo '{termo}'"] = re.compile(re.escape(termo), re.I)

    isentos = carregar_isencoes(raiz / cli.isencoes)
    achados = arquivos = binarios = 0

    for f in sorted(raiz.rglob("*")):
        if f.is_dir() or ".git" in f.parts:
            continue
        arquivos += 1
        if f.name in ISENTOS_POR_ARQUIVO:
            continue
        try:
            texto = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Binario nao e' varrido — e e' justamente onde um vazamento passa
            # despercebido. Um .safetensors carrega metadado, um .png carrega
            # EXIF. O scanner avisa em vez de fingir que conferiu.
            binarios += 1
            if not cli.quieto:
                print(f"  [binario, NAO varrido] {f.relative_to(raiz)}")
            continue

        linhas = texto.splitlines()
        for rotulo, rx in padroes.items():
            for m in rx.finditer(texto):
                n = texto[:m.start()].count("\n")
                trecho = linhas[n].strip()[:88] if n < len(linhas) else ""
                if any(x in trecho for x in isentos):
                    continue
                print(f"  !! {rotulo:24s} {f.relative_to(raiz)}:{n + 1}  {trecho}")
                achados += 1

    print(f"\n{arquivos} arquivos varridos"
          + (f", {binarios} binarios NAO varridos" if binarios else "")
          + " — " + ("LIMPO" if not achados else f"{achados} A RESOLVER"))
    if achados:
        print(f"Falso positivo legitimo? Ponha o trecho em "
              f"{cli.isencoes}, com o motivo ao lado.")
    return 1 if achados else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
