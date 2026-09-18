"""Quanto do lote e' padding, de verdade — e quanto um bucket por comprimento salva.

POR QUE MEDIR ANTES DE IMPLEMENTAR PACKING
------------------------------------------
Packing/varlen (juntar N exemplos num unico tensor `total_tokens x dim` e dizer
ao FlashAttention onde cada um termina) e' a maior obra de engenharia que resta
no caminho de aceleracao. So' vale se houver padding para eliminar.

E o padding daqui NAO e' o que se imagina: o `TextDataCollator` do vendor faz

    max_tokens_length = min(max(len(e) for e in batch), self.max_length)

ou seja, preenche ate' o MAIOR EXEMPLO DO LOTE, nao ate' `max_length`. Com
`--max-length 512` e mediana 145, quem manda e' o sorteio do lote, nao o teto.
O desperdicio e' entao uma propriedade da DISTRIBUICAO e do tamanho do lote, e
cresce com o batch: quanto mais amostras no lote, maior o maximo esperado.

O QUE ESTE SCRIPT RESPONDE
--------------------------
1. `tokens uteis / tokens processados` no lote como ele e' montado hoje;
2. quanto disso um BUCKET por comprimento recupera — a versao sem risco, que
   nao mexe no modelo, so' na ordem em que o dataloader entrega;
3. quanto sobraria para o packing de verdade ganhar depois do bucket.

    python scripts/medir_padding_fish.py --pack meu-pack
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def lote_a_lote(comprimentos: list[int], batch: int, teto: int) -> tuple[int, int]:
    """Soma tokens UTEIS e tokens PROCESSADOS, na ordem dada."""
    uteis = processados = 0
    for i in range(0, len(comprimentos) - batch + 1, batch):
        grupo = comprimentos[i:i + batch]
        largura = min(max(grupo), teto)
        uteis += sum(min(c, teto) for c in grupo)
        processados += largura * len(grupo)
    return uteis, processados


def _relatorio(cli, comprimentos: list[int]) -> int:
    if not comprimentos:
        raise SystemExit("nao consegui ler comprimento de nenhuma amostra")

    n = len(comprimentos)
    ordenado = sorted(comprimentos)
    print(f"pack {cli.pack}: {n} amostras")
    print(f"  mediana {ordenado[n//2]}  p90 {ordenado[int(n*.9)]}  "
          f"p95 {ordenado[int(n*.95)]}  p99 {ordenado[int(n*.99)]}  "
          f"max {ordenado[-1]}  media {st.mean(comprimentos):.0f}")

    rng = random.Random(cli.seed)
    linhas = []
    print(f"\n{'batch':>6} {'hoje':>9} {'com bucket':>12} {'recupera':>10} "
          f"{'sobra p/ packing':>17}")
    print("-" * 60)
    for b in [int(x) for x in cli.batches.split(",")]:
        embaralhado = comprimentos[:]
        rng.shuffle(embaralhado)
        u1, p1 = lote_a_lote(embaralhado, b, cli.max_length)

        # BUCKET: ordenar por comprimento antes de lotear. Cada lote fica com
        # exemplos parecidos, entao o maior do lote deixa de puxar os outros.
        # Os lotes continuam embaralhados ENTRE si — o que importa para o
        # treino e' que lotes consecutivos nao sejam todos curtos ou todos
        # longos, nao que o lote seja internamente heterogeneo.
        u2, p2 = lote_a_lote(sorted(embaralhado), b, cli.max_length)

        ef1, ef2 = u1 / p1, u2 / p2
        print(f"{b:>6} {ef1:8.1%} {ef2:11.1%} {ef2/ef1 - 1:+9.1%} "
              f"{1 - ef2:16.1%}")
        linhas.append({"batch": b, "eficiencia_hoje": ef1,
                       "eficiencia_bucket": ef2,
                       "ganho_bucket": ef2 / ef1 - 1, "sobra_packing": 1 - ef2})

    print("\n'hoje' = tokens uteis / tokens processados com lote aleatorio.")
    print("'com bucket' = o mesmo, agrupando por comprimento antes de lotear.")
    print("'sobra p/ packing' = o que o packing de verdade ainda teria a ganhar")
    print("                     DEPOIS do bucket — e' esse numero que decide se")
    print("                     a obra vale, nao o desperdicio bruto de hoje.")

    saida = ROOT / "reports" / "padding_fish.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(
        {"pack": cli.pack, "n": n, "max_length": cli.max_length,
         "mediana": ordenado[n // 2], "p95": ordenado[int(n * .95)],
         "max": ordenado[-1], "por_batch": linhas},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {saida}")


    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="pack")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batches", default="1,2,4,6,8")
    ap.add_argument("--amostras", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--exporters-root", default=None)
    ap.add_argument("--manifest", default=None,
                    help="le' os comprimentos de um manifest.jsonl em vez dos "
                         "protos. Serve para um pack que ainda NAO foi exportado "
                         "para o Fish: a eficiencia de padding depende da FORMA "
                         "da distribuicao, e o numero de frames semanticos e' "
                         "proporcional a' duracao (o texto acrescenta um offset "
                         "quase constante, que so' melhora a razao).")
    ap.add_argument("--frames-por-s", type=float, default=21.5,
                    help="frames semanticos por segundo; so' muda a escala, nao "
                         "as razoes que este script reporta")
    cli = ap.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "vendor" / "fish-speech"))
    from fish_speech.datasets.protos.text_data_pb2 import SampledData, TextData
    from fish_speech.datasets.protos.text_data_stream import read_pb_stream

    comprimentos: list[int] = []

    if cli.manifest:
        import json as _json

        with open(cli.manifest, encoding="utf-8") as f:
            for linha in f:
                d = _json.loads(linha)
                if d.get("duration"):
                    comprimentos.append(int(round(d["duration"] * cli.frames_por_s)))
                if len(comprimentos) >= cli.amostras:
                    break
        return _relatorio(cli, comprimentos)

    raiz = Path(cli.exporters_root) if cli.exporters_root else ROOT / "exporters"
    protos = sorted((raiz / "fish" / cli.pack / "protos").glob("*.protos"))
    if not protos:
        raise SystemExit(f"nenhum .protos em {raiz / 'fish' / cli.pack / 'protos'}")

    # O comprimento em TOKENS nao esta' gravado: o proto guarda texto e as VQ
    # codes. O que entra no modelo e' uma linha por token semantico mais o
    # texto, entao o numero de frames das VQ codes e' a parte que domina e a
    # unica que da' para ler sem rodar o tokenizador em 4.000 amostras.
    comprimentos: list[int] = []
    for arq in protos:
        with open(arq, "rb") as f:
            for item in read_pb_stream(f):
                for sentenca in item.sentences:
                    if sentenca.semantics and sentenca.semantics[0].values:
                        comprimentos.append(len(sentenca.semantics[0].values))
                if len(comprimentos) >= cli.amostras:
                    break
        if len(comprimentos) >= cli.amostras:
            break

    return _relatorio(cli, comprimentos)


if __name__ == "__main__":
    raise SystemExit(main())
