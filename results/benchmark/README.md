# Benchmark final

Este diretório contém os resultados de seis algoritmos nas 74 instâncias. Os dois shards representam trabalhos complementares, não duas cópias de uma mesma execução.

O índice unificado e auditado está em [`../tables/analysis_index_final.csv`](../tables/analysis_index_final.csv): filtre `phase=benchmark` para as 3.936 execuções. As 190 trajetórias complementares estão em `../stn/`, separadas dos resultados usados no ranking.

As entradas compartilhadas entre os laboratórios foram reunidas em [`../../data/instances/`](../../data/instances/). Protocolos e metadados por laboratório permanecem nos shards; caminhos históricos foram normalizados. Os artefatos referenciados por `source_artifact_local` são resolvidos a partir da raiz do repositório.

O campo bruto MIP de `job_03267` é 295; os campos analíticos do índice usam o objetivo 283 da solução validada. Duas falhas ILS permanecem identificadas. Veja [`../../docs/resultados.md`](../../docs/resultados.md).
