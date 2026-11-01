# Resultados e interpretação

A referência tabular é [`results/tables`](../results/tables/). A monografia discute os resultados no contexto das perguntas de pesquisa. Este guia destaca as escolhas de leitura que afetam a interpretação.

## Comparação dos seis algoritmos

O índice final tem 4.126 linhas: 3.936 com `phase=benchmark` e 190 com `phase=trajectory`. O ranking dos seis algoritmos usa somente as primeiras. As matrizes de objetivos e gaps têm 74 linhas e seis colunas de algoritmos.

O IBS apresenta o menor gap médio geral (3,48662%) e a melhor média, incluindo empates, em 44 instâncias. Isso não estabelece superioridade em todos os casos. CUTH tem 20 melhores médias e MIP tem 11; contagens incluem empates e podem somar mais de 74.

Há duas falhas ILS de solução final inviável. A política de análise atribui a elas o baseline viável sem recursos; o registro de falha permanece. Em `job_03267`, o objetivo bruto MIP é 295, enquanto a solução salva validada tem 283. Use `analysis_objective` ou `canonical_analysis_objective` no índice final. A coluna `objective` conserva o bruto.

Os gaps percentuais estão em `gap_percent_74x6.csv`; as entradas de desempenho da MATILDA usam frações. Confundir as duas unidades multiplica o limiar por 100.

## ISA e métricas direcionadas

Os modelos de controle e topologia oferecem representações relacionadas, mas com diferenças em seleção de atributos e vizinhanças. A contribuição topológica é principalmente estrutural e exploratória. Os efeitos nos diagnósticos preditivos dependem do algoritmo.

As métricas em amostra não medem generalização. A validação cruzada auxiliar usa repetições, mas não aninha todo o pipeline de seleção e representação, nem constitui validação externa. O desbalanceamento dos rótulos, especialmente do IBS, exige cuidado ao interpretar accuracy.

Os quatro modelos corrigidos estão em `results/isa/`. Os mapas de `results/generation/` são a referência fixa dos alvos da geração, não modelos concorrentes a serem usados para escolher retrospectivamente o melhor resultado.

## IBS e trajetórias

As 190 buscas decimais registraram 36.593 estados ao longo das trajetórias, com 2.682 estados distintos quando contados separadamente por instância. As STNs mostram diferenças de dispersão, compartilhamento de estados e concentração em soluções de boa qualidade.

O último estado RAW coincide com o objetivo do incumbente em 67 das 190 buscas. Nos outros casos, o último registro não é a melhor solução encontrada. As 190 buscas ultrapassaram o limite nominal; os tempos efetivos estão publicados.

A comparação com as trajetórias anteriores encontrou 58 objetivos novos menores, 47 iguais e 85 maiores, sob reavaliação decimal. Essa tabela é um resultado arquivado. As trajetórias antigas foram omitidas da edição pública; hardware, carga e duração também diferem, portanto a comparação não isola o efeito causal do leitor temporal.

`nend` conta assinaturas finais distintas que não são melhores, não falhas de execução. `best_strength_uncapped` pode superar 1; `best_strength_legacy_capped` é limitado a 1. Arestas filtradas não devem ser descritas como transições RAW.

## Expansão do espaço

As 36 buscas geraram 36 finalistas exatos e 19 aceitações distintas. As três sementes melhoraram as métricas de cobertura no mapa histórico. O efeito indica viabilidade de reduzir lacunas internas naquela representação; não demonstra cobertura de todos os incêndios reais nem que o método encontrou as instâncias mais difíceis possíveis.

A monografia também apresenta a cobertura da união das 19 instâncias, calculada a partir das aceitações das três sementes. Esse cálculo é uma agregação do experimento, não uma geração adicional.

## Escopo e limitações

A base tem 74 instâncias e inclui dependências estruturais entre casos que compartilham grafos. O BKV é uma referência observada. As análises são empíricas e exploratórias; não demonstram relações causais nem uma melhoria preditiva universal da topologia.

As STNs de literatura pertencem a um protocolo anterior. Resultados de geração, benchmark e trajetórias são camadas distintas e não devem ser combinados como repetições de um mesmo experimento.

Os arquivos de `results/audit/` e `results/monograph_audit/` registram verificações realizadas no estudo. Alguns relatórios preservam contagens e hashes do pacote original completo, que incluía artefatos não publicados aqui. A verificação desta edição utiliza o manifesto próprio de integridade e [`scripts/verify_repository.py`](../scripts/verify_repository.py).
