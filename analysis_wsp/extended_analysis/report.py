from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .provenance import write_json


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_Sem linhas mensuradas._"
    columns = [column for column in columns if column in frame]
    rows = frame[columns].fillna("").astype(str)
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows.to_numpy()],
        ],
    )


def generate_analysis_figures(result_root: Path) -> list[Path]:
    """Create the figure needed to audit retained topology attributes."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit = _csv(result_root / "02_instancespace_topology" / "topology_feature_audit.csv")
    if audit.empty or not {"family", "decision"}.issubset(audit):
        return []
    figures = result_root / "02_instancespace_topology" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    counts = audit.groupby(["family", "decision"]).size().unstack(fill_value=0).sort_index()
    axis = counts.plot(kind="bar", stacked=True, figsize=(10, 5.5), colormap="tab20")
    axis.set_xlabel("família topológica")
    axis.set_ylabel("atributos agregados")
    axis.set_title("Auditoria de atributos topológicos")
    axis.grid(axis="y", alpha=0.25)
    path = figures / "topology_feature_audit.png"
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(axis.figure)
    write_json(
        result_root / "analysis_figure_manifest.json",
        {"figures": [str(path.relative_to(result_root))], "result_provenance": "measured"},
    )
    return [path]


def write_report(root: Path, result_root: Path, config: Any, provenance: dict[str, Any]) -> Path:
    topology_validation = _json(result_root / "02_instancespace_topology" / "exact_ga_topology_recomputation_validation.json")
    projection_validation = _json(result_root / "05_final_isa" / "ga_projection_validation.json")
    terminal = _json(result_root / "terminal_acceptance_validation.json")
    embedding = _json(result_root / "05_final_isa" / "embedding_comparison_controle_vs_topologia.json")
    isa_comparison = _json(result_root / "05_final_isa" / "isa_artifact_comparison_controle_vs_topologia.json")
    reach = _json(result_root / "04a_ga_internal" / "collection_reach_summary.json")
    audit = _csv(result_root / "02_instancespace_topology" / "topology_feature_audit.csv")
    transformations = _csv(result_root / "02_instancespace_topology" / "topology_transformations.csv")
    ga = _csv(result_root / "04a_ga_internal" / "ga_target_results_with_wall_auc.csv")
    selected = _json(result_root / "02_instancespace_topology" / "threshold_10" / "sifted_selected_features.json")
    selected_features = [str(value) for value in selected.get("features", [])]
    selected_topology = [value for value in selected_features if value.removeprefix("feature_").startswith("topo_")]
    family = (
        audit.groupby(["family", "decision"]).size().unstack(fill_value=0).reset_index()
        if not audit.empty and {"family", "decision"}.issubset(audit)
        else pd.DataFrame()
    )
    accepted = int(ga.get("accepted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    seeds = sorted(pd.to_numeric(ga.get("root_seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique())
    execution_state = "concluída" if terminal.get("status") == "passed" else "incompleta"
    selected_topology_text = "\n".join(f"- `{value}`" for value in selected_topology) or "_Nenhuma feature topológica foi selecionada pelo SIFTED._"
    text = f"""# Complemento do TCC: topologia direcionada na MATILDA e povoamento WSP

**Estado da execução:** {execution_state}. Resultados só são finais quando as validações terminais estão aprovadas.

## Relação com o TCC já desenvolvido

A metodologia está documentada em `docs/metodologia.md`. Neste estudo, a ISA/MATILDA caracteriza as 55 instâncias WSP na pergunta macro “que tipo de problema é esta instância?” e as STNs descrevem a pergunta micro “como o IBS percorreu esta instância?”.

Esta execução acrescenta topologia direcionada à ISA e gera instâncias WSP sintéticas válidas para povoar lacunas internas. Ela não reescreve os relatórios, não recalcula STNs e não modifica dados ou resultados já concluídos.

## Objetivo e perguntas delimitadas

O experimento testa métricas de grafos direcionados na MATILDA e verifica se o gerador formal do WSP consegue produzir instâncias sintéticas válidas para povoar o espaço de busca. As famílias de métricas são descritas em `docs/metodologia.md`. A geração orientada a lacunas segue Smith-Miles e Bowly (2015, DOI `10.1016/j.cor.2015.04.022`); a alocação entre GA e busca local é uma adaptação inspirada em Bowly (2019, DOI `10.1145/3321707.3321877`), não uma reprodução literal do mecanismo paralelo daquele artigo.

As perguntas são: (1) as métricas podem ser extraídas e adicionadas de modo consistente à ISA?; (2) que dimensão estrutural adicional elas oferecem para caracterizar as instâncias?; e (3) o GA consegue preencher lacunas internas com instâncias válidas do domínio WSP? Não são objetivos desta execução estimar desempenho de algoritmos em instâncias novas, representar incêndios reais ou estender as STNs.

## Métricas, semântica e relevância

Cada linha da MATILDA é uma instância WSP. Métricas de nós são agregadas por média, desvio-padrão, mínimo, quartis, percentil 90 e máximo, transformando a topologia do dígrafo em atributos da instância.

| Família | O que acrescenta à discussão do TCC |
| --- | --- |
| Grau de entrada/saída, razão e desequilíbrio | Conectividade local e papéis de fonte/sumidouro; respeita a assimetria `u → v` versus `v → u` causada por vento e relevo. |
| PageRank | Influência estrutural global além das contagens locais de ligações. |
| HITS: hub e authority | Papéis complementares de disseminar para destinos relevantes e receber de emissores relevantes. |
| Betweenness direcionada | Gargalos e pontes em caminhos direcionados; qualquer relação com dificuldade ou trajetória IBS é apenas associativa. |
| Harmônica de entrada/saída | Acessibilidade e alcance assimétricos, definidos também para componentes desconectados. |
| Clustering direcionado | Coesão local, triângulos e rotas alternativas que complementam atributos temporais/dinâmicos e a leitura local das STNs. |

Tempos de propagação são usados como distância nos caminhos mínimos. Para PageRank e HITS ponderados, `1/tempo` é a força da ligação, evitando tratar arco lento como atalho. A transformação para suavizar caudas longas e padronizar escalas é implementada pelo PRELIM da MATILDA: deslocamento para valores positivos, Box–Cox e z-score, com parâmetros salvos em `prelim_feature_parameters.csv`. A projeção bidimensional final é o PILOT da MATILDA; não é executado um PCA separado.

O teste compara, no mesmo limiar de 10%, a entrada com os atributos já existentes (controle interno) e a mesma entrada acrescida da topologia. O GA varia somente parâmetros aceitos pelo gerador WSP. Instâncias aceitas são sintéticas, não observações reais, e não recebem rótulos de desempenho de algoritmos.

## Verificações de validade

| Verificação | Estado |
| --- | --- |
| Recomputação exata da topologia | `{topology_validation.get('status', 'missing')}` |
| Projeção dos candidatos no espaço ISA | `{projection_validation.get('status', 'missing')}` |
| Aceitação terminal | `{terminal.get('status', 'missing')}` |

Foram auditadas {len(audit)} features agregadas. A MATILDA selecionou {len(selected_features)} features, das quais {len(selected_topology)} são topológicas.

{_table(family, list(family.columns))}

Auditoria anterior ao PRELIM:

{_table(transformations, ['feature', 'sample_skewness', 'long_tail_flag_abs_skew_gt_2', 'transformation', 'matilda_prelim', 'result_provenance'])}

Features topológicas efetivamente selecionadas pelo SIFTED:

{selected_topology_text}

Comparação controle versus topologia:

| Resultado | Valor |
| --- | --- |
| Deslocamento alinhado médio do mapa | `{embedding.get('mean_aligned_displacement', 'missing')}` |
| Sobreposição média de vizinhos | `{embedding.get('mean_neighbor_overlap', 'missing')}` |
| Correlação de distâncias no mapa | `{embedding.get('pairwise_distance_spearman', 'missing')}` |
| Jaccard das features SIFTED | `{isa_comparison.get('sifted_jaccard', 'missing')}` |
| Concordância binária PYTHIA | `{isa_comparison.get('pythia_binary_agreement', 'missing')}` |

O GA registrou {len(ga)} buscas, {accepted} aceitações e seeds {seeds}. Após deduplicação por conteúdo, a união contém `{reach.get('deduplicated_seed_union', 'missing')}` instâncias aceitas distintas. Zero aceitações continua sendo um resultado negativo válido do experimento, desde que todos os orçamentos e validações tenham terminado. As fontes numéricas são `02_instancespace_topology/` e `04a_ga_internal/` no diretório `{result_root.relative_to(root)}`.

## Interpretação e uso na redação

As conclusões devem se limitar às métricas testadas na MATILDA e à geração de instâncias sintéticas válidas para lacunas internas. Não interpretar as instâncias como incêndios reais nem como evidência de desempenho de algoritmos. Relatar separadamente o que já estava demonstrado nos relatórios ISA/STN e o que esta extensão mediu sobre topologia e cobertura.

Para reproduzir em outra máquina, usar somente `analysis_wsp/run_extended_analysis.py --profile tcc` conforme `analysis_wsp/GUIA_EXECUCAO_TCC.md`. Os diretórios de resultados e a validação terminal são a fonte dos números deste relatório; os relatórios originais e seus dados permanecem como base consultável.

O manifesto de proveniência cobre {provenance.get('artifact_files_covered', 0)} artefatos. `sha256_manifest.json` fecha a integridade da execução.
"""
    path = result_root / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path
