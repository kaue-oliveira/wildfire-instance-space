# Monografia

Texto do TCC **Caracterização e expansão do espaço de instâncias do Wildfire Suppression Problem**, de Kauê de Oliveira Silva, UFLA.

- [PDF do trabalho](templufla_main.pdf).
- [Arquivo principal LaTeX](templufla_main.tex).
- [Seções do texto](secoes/).
- [Referências bibliográficas](refbib.bib).
- [Figuras dos resultados](imgs/resultados/).

Compile nesta pasta com `latexmk -pdf -interaction=nonstopmode templufla_main.tex`. A compilação usa BibTeX e os pacotes declarados no preâmbulo e na classe `templufla.cls`.

O texto usa o template acadêmico da UFLA. A origem e os créditos estão em [THIRD_PARTY.md](../THIRD_PARTY.md) e nos cabeçalhos dos arquivos de classe e estilos. Os recálculos associados aos resultados estão em `../results/monograph_audit/`.
