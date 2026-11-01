# Reprodução e verificação

Execute os comandos a partir da raiz do repositório. A conferência do pacote não executa os algoritmos WSP.

## 1. Conferir os arquivos publicados

```bash
python3 scripts/verify_repository.py
```

A rotina usa apenas a biblioteca padrão. Verifica hashes do manifesto desta edição, os artefatos do índice, 74 instâncias em dois formatos, as fases do índice e as dimensões das tabelas. Para recalcular objetivos, resumos e testes estatísticos a partir das soluções salvas:

```bash
python3 scripts/audit_benchmark.py
```

Os recálculos vão para `experiments/verification/`. As auditorias antigas são registros do estudo e podem mencionar arquivos intermediários omitidos desta edição.

## 2. Instalar o ambiente científico

O ambiente de referência é Python 3.12. As dependências diretas estão fixadas em `analysis_wsp/requirements-isa.txt`; a implementação pyInstanceSpace é obtida de uma revisão Git específica.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Os testes não executam uma campanha completa de solvers. O submódulo externo deve estar inicializado:

```bash
git submodule update --init --recursive
```

## 3. Reconstruir as STNs decimais

```bash
python scripts/rebuild_stns.py
```

A rotina lê as 190 trajetórias publicadas, reconstrói as redes RAW e filtradas e confere seus descritores com as 38 linhas agregadas da tabela final. Grava JSON, GraphML e tabelas de nós e arestas em `experiments/stn_rebuilt/`. Não executa o IBS. As figuras finais podem ser consultadas em `results/figures/stns19/`.

## 4. Reajustar os quatro modelos ISA

As matrizes atuais já estão publicadas. Para realizar novos ajustes sem substituir os resultados existentes, copie somente as entradas para a pasta de novos experimentos:

```bash
python scripts/prepare_isa_rerun.py
python -m analysis_wsp.matilda74.run_corrected_batch
```

Os ajustes usam SIFTED corrigido e gravam em `experiments/isa_corrected/`. Essa etapa treina modelos e pode demorar. Não reexecuta os seis algoritmos WSP nem a geração de instâncias. O diagnóstico auxiliar de validação cruzada não é uma validação externa do pipeline inteiro.

Para reextrair as matrizes a partir dos dados, existem os módulos `prepare`, `prepare_topology` e `prepare_comparison` em `analysis_wsp.matilda74`. A extração topológica exata de grafos grandes é custosa. Os novos arquivos de preparação vão para `experiments/isa_preparation/`.

## 5. Reexecutar a geração orientada

```bash
python -m analysis_wsp.run_extended_analysis --profile tcc --output-profile reproduction
```

A saída vai para `experiments/generation/reproduction/`. O protocolo tem três sementes-raiz, 36 alvos e 72.000 avaliações. Pode demandar muitas horas. A edição pública conserva os finalistas e curvas, mas não os arquivos individuais de todas as candidatas; inicie uma nova execução em vez de tentar retomar o arquivo histórico reduzido.

## 6. Preparar uma nova campanha do benchmark

São necessários compilador C++20, CMake, Boost, Gurobi e `gurobipy`, além do submódulo LBBD/ILS. A instalação e a licença do solver são locais e não fazem parte do repositório. As dependências particulares das implementações externas estão em [THIRD_PARTY.md](../THIRD_PARTY.md).

Exemplos de inspeção e preparação, usando uma licença local:

```bash
python -m analysis_wsp.benchmark74.run --profile reproduction_v2_4 doctor --license /caminho/local/gurobi.lic --instrument-ils
python -m analysis_wsp.benchmark74.run --profile reproduction_v2_4 prepare --license /caminho/local/gurobi.lic --shard-count 2 --official
python -m analysis_wsp.benchmark74.run --profile reproduction_v2_4 --help
```

Esses comandos usam `experiments/benchmark/reproduction_v2_4/`. A etapa de preparação também registra jobs históricos de trajetória; para reconstruir as redes decimais publicadas, use a etapa 3. Uma reexecução do benchmark exige repetir hardware, limites, sementes, versões e política de falhas para permitir comparação adequada. Não basta misturar novas medições às tabelas prontas.

## 7. Compilar a monografia

```bash
cd monografia
latexmk -pdf -interaction=nonstopmode templufla_main.tex
```

Use uma distribuição TeX com os pacotes requeridos pela classe UFLA, suporte a português e BibTeX. O PDF publicado corresponde ao texto disponível nesta edição. A compilação não depende dos materiais de revisão removidos.
