# Tabelas finais e unidades

| Arquivo | Unidade/uso |
| --- | --- |
| `analysis_index_final.csv` | 4.126 jobs; filtrar `phase=benchmark` para 3.936; objetivo analítico corrigido e caminhos locais |
| `per_instance_algorithm_decimal.csv` | 444 células; médias/desvios por instância e algoritmo, repetições e falhas |
| `algorithm_summary_decimal.csv` | 42 linhas: seis algoritmos × sete recortes; gap em **%**, peso igual por instância |
| `objective_mean_74x6.csv` | Matriz de objetivos médios; menor é melhor |
| `gap_percent_74x6.csv` | Matriz de gaps em **%**; a entrada MATILDA preservada usa **fração**, ou %/100 |
| `instance_catalog_74.csv` | Coleção, tamanho, vértices, seeds do benchmark e seeds complementares |
| `algorithm_failures_final.csv` | Duas falhas ILS; baseline incluído em objetivo, sem promoção a sucesso |
| `bkv_comparison.csv` | 74 referências históricas/atuais/seis algoritmos; todas sem mudança |
| `paired_instance_comparisons_decimal.csv` | 15 pares; teste de sinais por instância e correção Holm |
| `run_audit.csv` | 190 buscas novas; incumbente/último estado, duração, limite, validade e hash |
| `stn_descriptors.csv` | 38 agregados: 19 × RAW/filtro; coluna `representation` obrigatória |
| `stn_per_seed_descriptors.csv` | 380 redes individuais, nas duas representações |
| `legacy_vs_decimal_quality.csv` | 190 pares; novo melhor versus melhor decimal entre estados históricos arquivados |
| `legacy_vs_decimal_metrics.csv` | Métricas antigas e novas; fontes de trajetórias/protocolos distintos |
| `isa_stn_relation.csv` | 76 cruzamentos: 19 × 2 representações × 2 modelos ISA |
| `stn_performance_correlations.csv` | Spearman exploratório de 19 RAW contra gap IBS do benchmark; p nominal |
| `isa_selected_features_40.csv` | 10 seleções de cada um dos quatro modelos atuais |
| `isa_labels_and_actual_folds_24.csv` | Rótulos a 10% e folds efetivamente usados no diagnóstico auxiliar |
| `isa_in_sample_diagnostics_24.csv` | Diagnóstico em amostra, sem interpretação como generalização |
| `isa_auxiliary_cv_summary_24.csv` | Dez repetições de CV auxiliar, sem aninhamento do pipeline completo |
| `isa_footprints_24.csv` | Footprints por modelo/algoritmo; conferir área, densidade e pureza separadamente |
| `stn_literature16_historical_filtered.csv` | 16 STNs históricas filtradas; não são RAW nem novas execuções |
| `ibs_literature16_historical_runs.csv` | 160 registros de literatura; protocolo histórico |
| `geracao_ga/` | Alvos, convergência, alcance e aceitação medidos no mapa de geração histórico |

`objective` no índice é o valor bruto. `analysis_objective` e
`canonical_analysis_objective` no **índice final** são o valor auditado: 283 no
MIP `job_03267`. Os resultados individuais conservam o objetivo bruto para rastreabilidade; o índice final é a fonte analítica prioritária.

`nend` é a quantidade de assinaturas finais distintas não melhores. Não é número
de falhas. `best_strength_uncapped` pode superar 1; a versão `legacy_capped`
é limitada. `shared_node_fraction_legacy` e `mean_pairwise_jaccard` têm definições
diferentes. Tabelas de STN nunca substituem o status de sucesso do benchmark.
