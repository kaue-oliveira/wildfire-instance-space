# Código externo e créditos

As implementações externas permanecem identificadas e separadas em `third_party/`. Seus arquivos de licença e avisos de autoria foram preservados. Este repositório não atribui uma licença única a materiais de origens diferentes.

| Diretório | Origem | Uso |
| --- | --- | --- |
| `third_party/Wildfire-Suppression/` | https://github.com/gutodelazeri/Wildfire-Suppression | Implementação C++ de IBS/MIP/CUTH/RS, gerador e instâncias originais |
| `third_party/Iterated-Beam-Search/` | https://github.com/gutodelazeri/Iterated-Beam-Search | IBS e suporte às trajetórias; licença preservada no diretório |
| `third_party/Logic-based-Benders-decomposition-for-wildfire-suppression/` | https://github.com/mitchopt/Logic-based-Benders-decomposition-for-wildfire-suppression | LBBD, ILS e MIP; submódulo na revisão `7a7b76ff03548f622c41308cff60ed65451bf0b5` |
| `third_party/WSP/` | https://github.com/gutodelazeri/WSP | Referência de gerador e dados; licença MIT preservada |
| `third_party/STNs-v2/` | Implementação STNs-v2 distribuída com o estudo; origem indicada no cabeçalho dos scripts | Scripts R de referência e licença preservada |
| `monografia/templufla.cls` e estilos | https://github.com/joaopaulo7/Template-Trabalhos-Academicos-UFLA | Template acadêmico UFLA e estilos abnTeX2; créditos nos próprios arquivos |

O código específico de análise deste trabalho está em `analysis_wsp/` e `scripts/`. O TCC e os resultados têm autoria indicada no README e em `CITATION.cff`. A publicação não acrescenta uma licença geral de redistribuição ao trabalho; consulte os avisos aplicáveis a cada componente antes de redistribuí-lo.

## Dependências dos solvers

- Wildfire-Suppression: C++20, CMake, Boost e Gurobi.
- Iterated-Beam-Search: C++ e Boost.
- LBBD e MIP de Harris et al.: Python, NumPy e Gurobi; ILS tem dependências próprias da implementação.
- STNs-v2: R e pacotes informados pelos scripts.

As instâncias suplementares são publicadas em dois formatos. Use o arquivo correspondente à implementação: o esquema nativo não é intercambiável diretamente com os campos `Nodes`, `Arcs`, `Delay`, `ArrivalTimeTarget` e `ResAtTime` da representação Harris.
