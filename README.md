# Gema EDA

Aplicativo didático de Análise Exploratória de Dados (EDA) com base nos passos de John W. Tukey.

## Como executar localmente

```powershell
python -m streamlit run app.py
```

O app aceita arquivos CSV. Se nenhum arquivo for enviado, ele usa uma base exemplo de atendimentos para demonstrar o fluxo completo.

## Fluxo contemplado

1. Carga de uma base de dados CSV.
2. Análise exploratória com estrutura, qualidade, medidas resistentes, histogramas, boxplots, outliers, grupos e tempo.
3. Explicação didática dos achados, incluindo Random Forest e R².
4. Download do relatório final em PDF.

## Deploy no Streamlit Community Cloud

Use estes parâmetros:

- Repository: `CarolRodrigues-SEPLAG/EDA`
- Branch: `main`
- Main file path: `app.py`
