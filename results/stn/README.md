# Trajetórias decimais e Search Trajectory Networks

`trajectories/` contém 190 execuções IBS: 19 instâncias, dez sementes cada. Há trajetória RAW, solução incumbente, resultado e registro operacional por execução.

`stns/aggregate/` contém 38 redes agregadas (19 × RAW/filtro); `stns/per_seed/`, 380 redes individuais. JSON, GraphML e CSV descrevem os mesmos nós e arestas. As figuras selecionadas estão em `../figures/stns19/`.

Os descritores finais estão em `../tables/stn_descriptors.csv` e `../tables/stn_per_seed_descriptors.csv`. O índice global separa estes registros com `phase=trajectory`; eles não são repetições adicionais do benchmark.

A reconstrução pode ser executada sem solver com `python scripts/rebuild_stns.py`, a partir da raiz do projeto. A [metodologia](../../docs/metodologia.md) explica a diferença entre arestas RAW e filtradas.
