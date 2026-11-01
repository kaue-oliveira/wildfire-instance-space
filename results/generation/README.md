# Geração orientada no mapa histórico

Os diretórios numerados preservam a correspondência com o pipeline:

- `00_baseline_55/`: base e verificações das 55 instâncias originais.
- `01_instancespace_original/`: representação histórica de controle.
- `02_instancespace_topology/`: métricas auditadas e mapa topológico usado na geração.
- `04a_ga_internal/`: alvos, curvas por semente, resultados e finalistas.

Os mapas aqui são necessários para interpretar os alvos, distintos dos quatro modelos atuais em `../isa/`.

O arquivo `04a_ga_internal/ga_target_results.csv` identifica 36 buscas e as 19 instâncias aceitas distintas. Os subdiretórios `seed_*/generated_candidates/` conservam os finalistas exatos, suas instâncias e registros de características selecionadas. As curvas preservam o percurso das 72.000 avaliações; os 72 mil JSONs intermediários foram omitidos.

A coleção de entrada do benchmark em `../../data/instances/` contém os arquivos preparados e validados para os solvers. Eles podem diferir dos finalistas nativos no campo de contagem de recursos, conforme documentado na [metodologia](../../docs/metodologia.md).
