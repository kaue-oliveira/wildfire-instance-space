# Implementações externas

As origens, dependências e avisos de licença estão em [THIRD_PARTY.md](../THIRD_PARTY.md).

Inicialize a implementação LBBD/ILS fixada como submódulo:

```bash
git submodule update --init --recursive
```

As 74 instâncias usadas no benchmark estão em `../data/instances/`, em formatos nativo e Harris. Novas compilações e resultados devem permanecer locais. Os READMEs de cada implementação descrevem seu uso particular.
