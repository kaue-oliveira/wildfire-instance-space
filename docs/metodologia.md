# Problema e metodologia

## Perguntas de pesquisa

1. Quais características das instâncias se relacionam ao desempenho dos algoritmos e o que a topologia direcionada acrescenta?
2. Como o comportamento de busca do IBS varia entre instâncias?
3. É possível produzir instâncias válidas que ocupem regiões pouco representadas no espaço observado?

O [texto completo](../monografia/templufla_main.pdf) apresenta a fundamentação e o protocolo. Este documento orienta a leitura do código e dos dados.

## Instâncias e objetivo

A base original reúne 16 instâncias de literatura (`LA0`–`LA7`, `LB0`–`LB7`) e 39 produzidas pelo gerador WSP. A geração orientada acrescenta 19 casos, totalizando 74.

Cada arquivo nativo descreve vértices, arcos e tempos, ignições, horizonte e recursos. A conversão para o formato Harris atende às implementações LBBD/ILS. As duas representações estão em [`data/instances`](../data/instances/). Na preparação do benchmark, quatro instâncias suplementares tiveram o campo de quantidade de recursos alinhado ao comprimento dos vetores; os arquivos finalistas da geração preservam a representação anterior e sua validação.

O objetivo é contar os vértices atingidos antes do horizonte. A chegada exatamente no horizonte não entra nessa contagem. A supressão modifica a propagação conforme o modelo temporal da instância. Os validadores utilizam os tempos em centésimos inteiros para evitar truncamento.

## Benchmark

Comparam-se CUTH, IBS, ILS, LBBD, MIP e RS nas mesmas 74 instâncias. Cada algoritmo tem 10 sementes por instância Small, Medium e Large, e 3 nas Huge.

| Classe | Instâncias | Sementes por algoritmo | Limite nominal (s) |
| --- | ---: | ---: | ---: |
| Small | 20 | 10 | 600 |
| Medium | 33 | 10 | 1.350 |
| Large | 9 | 10 | 2.400 |
| Huge | 12 | 3 | 9.600 |

O total é 656 execuções por algoritmo e 3.936 no benchmark. O desempenho é agregado primeiro por instância. O gap percentual é `100 × (objetivo − BKV) / BKV`, com BKV igual ao melhor valor observado. Os limites são nominais: rotinas que verificam o relógio entre chamadas completas podem ultrapassá-los. Tempos efetivos e protocolos estão registrados por execução.

## ISA e topologia

A unidade de análise da ISA é uma instância. Características de nós são agregadas para produzir descritores por instância; resultados por nó não se tornam observações independentes de desempenho.

As famílias topológicas incluem grau de entrada/saída, razões e desequilíbrios, PageRank, HITS, betweenness direcionada, centralidade harmônica de entrada/saída e clustering direcionado. As agregações são média, desvio-padrão, mínimo, quartis 25/50/75, percentil 90 e máximo. Foram avaliados 136 descritores topológicos; a auditoria histórica reteve 73 após remover constantes, redundâncias e instabilidades.

Tempos de propagação representam distâncias em caminhos mínimos. Nas versões ponderadas de PageRank e HITS, a força é o inverso do tempo. Os descritores são enviados à MATILDA sem transformação manual adicional: PRELIM aplica deslocamento positivo, Box–Cox e padronização; SIFTED seleciona características; PILOT constrói a projeção bidimensional.

O limiar de bom desempenho é 10%. Os quatro modelos publicados usam a correção SIFTED de direção da aptidão, filtro de correlação e correspondência entre colunas avaliadas e retornadas. Há controle e topologia para 74 instâncias e, sob protocolo comparável, para as 55 originais. Os mapas que orientaram a geração são preservados separadamente em `results/generation`: seus alvos dependem daquela representação fixa.

## Trajetórias e STNs

A campanha complementar executa o IBS em 19 instâncias × 10 sementes, preservando tempos decimais. Essas 190 buscas ficam separadas do benchmark. As 16 STNs de literatura são históricas e não representam reexecuções decimais novas.

Na representação RAW, arestas correspondem a transições observadas dentro de uma semente. No filtro histórico de não piora, mantêm-se registros com objetivo não superior ao melhor anterior; uma aresta filtrada pode conectar estados não consecutivos na trajetória original. Não são criadas transições entre sementes.

Para as instâncias suplementares, há 190 redes por representação, mais 19 agregadas por representação: 418 redes no total. JSON, GraphML e tabelas de nós/arestas permitem leitura independente dos layouts gráficos.

## Geração orientada

A geração identifica alvos em lacunas internas do mapa histórico e adapta o gerador WSP por um algoritmo genético com busca local. Usa três sementes-raiz (`20260612`, `104729`, `1299709`), 12 alvos por semente e 2.000 avaliações por alvo, totalizando 72.000 avaliações.

Cada busca destina 1.600 avaliações ao GA e 400 à busca local. Os 36 finalistas passam por recomputação exata. A aceitação exige melhora da distância ao alvo e pertinência ao fecho empírico e à região válida da MATILDA. A coleção final tem 19 instâncias distintas aceitas.

O repositório preserva curvas, finalistas, metadados e mapas necessários à análise. Os arquivos individuais das candidatas intermediárias e caches completos de execução foram omitidos; a publicação não permite retomar aquela execução a partir de cada avaliação intermediária.

## Referências metodológicas

- Smith-Miles e Bowly (2015), *Generating new test instances by evolving in instance space*. DOI: `10.1016/j.cor.2015.04.022`.
- Bowly (2019), *A Genetic Algorithm Hybridisation Scheme for Effective Use of Parallel Workers*. DOI: `10.1145/3321707.3321877`.
- As referências da ISA, STNs, modelos WSP e métricas direcionadas constam na [bibliografia da monografia](../monografia/refbib.bib).
