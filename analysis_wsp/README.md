# Código de análise do WSP

O [mapa do repositório](../docs/estrutura.md) descreve os módulos, e o [guia de reprodução](../docs/reproducao.md) apresenta os comandos.

- `extended_analysis/`: extração, topologia, ISA e geração orientada.
- `benchmark74/`: protocolo e execução da comparação dos seis algoritmos.
- `matilda74/`: preparação e ajuste dos modelos com correção SIFTED.
- `ibs_decimal_pipeline/`: avaliação temporal e redes RAW/filtradas.
- `caminho_c/`: conversão e validação compartilhadas pelos experimentos.
- `tests/`: verificações científicas e de protocolo.

Os dados finais estão em `../results/`. A pasta `data/` deste pacote contém as entradas históricas das análises iniciais, incluindo as trajetórias de literatura.
