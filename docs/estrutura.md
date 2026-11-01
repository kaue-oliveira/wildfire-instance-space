# Organização do repositório

## Código

| Diretório | Responsabilidade |
| --- | --- |
| `analysis_wsp/extended_analysis/` | Características, topologia, ISA, geração e validações |
| `analysis_wsp/benchmark74/` | Preparação, execução e integração do benchmark de seis algoritmos |
| `analysis_wsp/caminho_c/` | Utilitários de conversão de instâncias, protocolo e validação compartilhados |
| `analysis_wsp/matilda74/` | Matrizes 55/74, ajustes e correção SIFTED |
| `analysis_wsp/ibs_decimal_pipeline/` | Modelo temporal em centésimos e construção de redes RAW/filtradas |
| `analysis_wsp/tests/` | Testes de protocolo, conversão, topologia, geração e correção SIFTED |
| `scripts/` | Verificação do pacote e reconstrução de STNs publicadas |

`analysis_wsp/data/` contém entradas da análise das 55 instâncias e trajetórias de literatura. O ponto de entrada para os resultados finais é `results/tables/`, não as tabelas preliminares.

## Dados e resultados

| Diretório | Conteúdo |
| --- | --- |
| `data/instances/native/` | Coleção canônica de 74 instâncias no formato nativo |
| `data/instances/harris/` | As mesmas instâncias convertidas para as implementações de Harris et al. |
| `results/benchmark/shards/` | Resultados individuais, soluções, logs e protocolos dos dois laboratórios |
| `results/isa/` | Quatro modelos atuais com SIFTED corrigido e suas matrizes |
| `results/isa_preparation/` | Características e cache topológico que apoiam a preparação da ISA |
| `results/stn/` | 190 trajetórias decimais e 418 redes correspondentes |
| `results/stn_literature/` | Redes de literatura em RData, quando disponíveis |
| `results/generation/` | Mapas de geração, alvos, curvas e 36 finalistas exatos |
| `results/tables/` | Tabelas finais para consulta e análise |
| `results/figures/` | Seleção de figuras de resultados |
| `results/audit/` | Registros de auditoria e normalização de metadados |
| `results/monograph_audit/` | Recálculos associados às tabelas e afirmações da monografia |

Nos arquivos de geração, os diretórios numerados identificam etapas do pipeline: base, ISA original, ISA topológica e geração. Os nomes de sementes e alvos foram preservados para manter a correspondência com as curvas e os manifestos.

## Texto do trabalho

`monografia/` contém o PDF e os arquivos necessários à compilação do TCC. A classe e os estilos LaTeX foram mantidos com seus créditos. Revisões de orientação e documentos de preparação pessoal não integram esta publicação.

## Proveniência e metadados

Objetivos, tempos medidos, sementes, soluções e trajetórias foram preservados. Caminhos absolutos das máquinas e rótulos de branch foram normalizados em metadados operacionais. Os hashes do índice foram atualizados para os arquivos públicos; os hashes anteriores dos metadados alterados estão em `results/audit/metadata_normalization.json`.

Os caminhos `source_artifact_local` e `source_run_directory_local` do índice final são relativos à raiz deste repositório. Outros manifestos registram o contexto original de execução; tokens como `{PROJECT_ROOT}` e `{USER_HOME}` indicam caminhos históricos e não devem ser usados como comandos.

A edição pública usa um histórico Git próprio. Os arquivos de compilação, ambientes virtuais, credenciais e saídas de novas execuções são ignorados pelo Git.
