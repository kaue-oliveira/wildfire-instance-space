# Contribuições

Abra uma issue para relatar erros de execução, inconsistências nos dados ou dúvidas sobre a documentação. Inclua o comando, a versão do Python, o sistema operacional e o erro observado. Remova credenciais e caminhos pessoais dos exemplos.

Para alterações de código, descreva o problema e o comportamento resultante, e execute `python -m pytest -q`. Se mudar caminhos, índices ou documentação, execute também `python3 scripts/verify_repository.py`.

Os resultados medidos publicados são uma referência fixa. Novas execuções devem ir para `experiments/`, com parâmetros, sementes, versões e hardware registrados. Uma correção de dado precisa indicar a fonte, a justificativa e o efeito nas tabelas; não substitua valores medidos por previsões.

Não inclua arquivos `.env`, chaves, tokens, licenças do solver, ambientes virtuais, caches ou executáveis compilados. Antes de fazer commit, confira `git diff --cached --stat` e `git diff --cached`.
