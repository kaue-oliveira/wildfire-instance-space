# Modelos ISA com SIFTED corrigido

Há quatro ajustes: controle e topologia nas 74 instâncias, e controle e topologia nas 55 originais sob protocolo comparável. Os dois últimos estão em `comparison_55_v24/`.

Cada modelo preserva coordenadas PILOT, atributos selecionados, parâmetros, predições PYTHIA, resultados TRACE e objeto serializado. As entradas utilizadas estão em `inputs/`. Os objetos `.pkl` exigem o ambiente científico de referência; as tabelas CSV permitem inspeção independente.

O código da correção está em `../../analysis_wsp/matilda74/sifted_correction.py`. Veja a [metodologia](../../docs/metodologia.md) e o [guia para novos ajustes](../../docs/reproducao.md). As comparações registradas com a implementação anterior são resultados históricos; os ajustes antigos foram omitidos.
