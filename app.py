from __future__ import annotations

# ============================================================
# 1) IMPORTAÇÕES E CONFIGURAÇÃO GERAL
# ============================================================
# Este bloco concentra as bibliotecas usadas no app. A ideia é manter o script
# em um único arquivo didático: quem abrir o código consegue ver a jornada
# inteira da EDA sem precisar navegar por vários módulos.

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# Configuração inicial da página Streamlit. Ela define título, layout largo e
# barra lateral aberta, porque o menu lateral é parte importante do roteiro.
st.set_page_config(
    page_title="Gema EDA | Roteiro de Tukey",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2) ESTILO VISUAL E TEXTOS FIXOS DO ROTEIRO
# ============================================================
# O CSS abaixo dá uma aparência mais organizada à aplicação sem criar uma
# dependência externa. As classes são usadas nos blocos didáticos da tela.

APP_CSS = """
<style>
    :root {
        --ink: #172033;
        --muted: #64748b;
        --line: #d8e0ea;
        --paper: #f8fafc;
        --panel: #ffffff;
        --accent: #1d4ed8;
        --accent-strong: #1e3a8a;
        --accent-soft: #e8f1ff;
        --warn-soft: #fff7df;
        --bad-soft: #ffe8ec;
    }

    .main .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.4rem;
        max-width: 1280px;
    }

    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f5f8fc 0%, #ffffff 36%);
    }

    [data-testid="stSidebar"] {
        background: #eef3f8;
        border-right: 1px solid var(--line);
    }

    .intro {
        border-left: 5px solid var(--accent);
        background: var(--panel);
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: .5rem 0 1rem 0;
        color: var(--ink);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }

    .eda-note {
        background: var(--accent-soft);
        border: 1px solid #b9d4ff;
        border-radius: 8px;
        padding: .85rem 1rem;
        color: var(--ink);
    }

    .eda-warning {
        background: var(--warn-soft);
        border: 1px solid #ead18c;
        border-radius: 8px;
        padding: .85rem 1rem;
        color: var(--ink);
    }

    .eda-risk {
        background: var(--bad-soft);
        border: 1px solid #f6b4bd;
        border-radius: 8px;
        padding: .85rem 1rem;
        color: var(--ink);
    }

    .small-muted {
        color: var(--muted);
        font-size: .92rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .65rem .85rem;
        background: var(--panel);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.035);
    }

    div[data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--line);
    }

    button[kind="primary"], div.stDownloadButton button {
        border-radius: 8px;
        border-color: var(--accent);
        background: var(--accent);
        color: white;
    }
</style>
"""


TUKEY_EXPLANATIONS = {
    "estrutura": (
        "Nesta etapa, fazemos a primeira leitura da base: quantas linhas existem, quantas colunas, "
        "quais são os tipos de dados, onde há valores ausentes e se existem duplicidades. "
        "Isso responde uma pergunta simples e essencial: posso confiar nesta base para começar a análise?"
    ),
    "numeric": (
        "O resumo numérico descreve o comportamento das variáveis quantitativas. A média é a soma dos "
        "valores dividida pela quantidade de registros, mas pode ser influenciada por extremos. A mediana "
        "é o valor do meio quando os dados são ordenados, por isso costuma representar melhor o valor típico. "
        "Q1 é o primeiro quartil, onde 25% dos dados ficam abaixo; Q3 é o terceiro quartil, onde 75% ficam abaixo. "
        "O IQR, ou intervalo interquartil, é Q3 - Q1 e mostra a faixa dos 50% centrais."
    ),
    "visual": (
        "Histogramas mostram como os valores se distribuem: concentração, caudas, assimetria e possíveis picos. "
        "Boxplots resumem mediana, quartis, dispersão e possíveis outliers. Juntos, esses gráficos ajudam a "
        "enxergar o padrão antes de tentar explicar o porquê."
    ),
    "outlier": (
        "Outliers são valores muito diferentes do comportamento central da base. Pelo critério de Tukey, "
        "marcamos valores abaixo de Q1 - 1,5 x IQR ou acima de Q3 + 1,5 x IQR. Eles não são automaticamente "
        "erros: podem ser casos raros, problemas de cadastro, exceções operacionais ou sinais importantes."
    ),
    "groups": (
        "Comparar grupos significa observar se o indicador principal muda entre categorias, como região, canal, "
        "produto, equipe ou perfil. Essa leitura transforma uma média geral em decisão prática, porque mostra "
        "onde a gestão pode priorizar investigação, recursos ou planos de ação."
    ),
    "time": (
        "A análise temporal mostra se o indicador está aumentando, diminuindo ou permanecendo estável ao longo "
        "do tempo. Para gestão, isso ajuda a separar um problema pontual de uma mudança persistente e indica "
        "se a ação deve ser corretiva, preventiva ou apenas monitorada."
    ),
    "model": (
        "Random Forest é um modelo que combina várias árvores de decisão. Cada árvore tenta prever o indicador "
        "principal usando os demais campos da base; depois, o conjunto das árvores vota ou tira uma média. "
        "Aqui ela é usada para priorizar investigação: quais campos ajudam mais a prever a variável escolhida. "
        "Isso indica influência preditiva, não prova causa e efeito."
    ),
}


EDA_DEFINITION = (
    "EDA é a sigla para Exploratory Data Analysis, ou Análise Exploratória de Dados. "
    "É a etapa em que investigamos a base antes de tomar decisões estatísticas ou criar modelos: "
    "olhamos qualidade, padrões, distribuições, diferenças entre grupos, valores extremos e hipóteses iniciais."
)


TUKEY_DEFINITION = (
    "John Wilder Tukey foi um estatístico norte-americano muito influente no século XX. "
    "Ele popularizou a ideia de explorar visualmente os dados antes de partir para testes formais, "
    "e também difundiu ferramentas como boxplot, quartis e análise resistente a valores extremos."
)


# ============================================================
# 3) ESTRUTURA DE CONTEXTO
# ============================================================
# Esta classe guarda as escolhas centrais do usuário. Passar esse objeto entre
# funções deixa o código mais claro do que repetir muitos parâmetros soltos.

@dataclass
class EDAContext:
    data: pd.DataFrame
    question: str
    unit: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    selected_numeric: str | None
    selected_group: str | None
    selected_time: str | None


# ============================================================
# 4) CARGA E PREPARAÇÃO DA BASE
# ============================================================
# As funções deste bloco cuidam da entrada dos dados, da base exemplo e de uma
# limpeza leve. Elas não fazem transformações agressivas, porque EDA robusta
# deve preservar a base até que exista uma justificativa clara para alterá-la.

def build_example_dataset(rows: int = 420) -> pd.DataFrame:
    """Cria uma base exemplo com nulos, duplicidades e outliers propositais."""
    rng = np.random.default_rng(42)
    regions = np.array(["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"])
    channels = np.array(["Loja física", "Aplicativo", "Site", "Telefone"])
    products = np.array(["Vacina A", "Vacina B", "Vacina C", "Vacina D"])
    dates = pd.date_range("2025-01-01", periods=180, freq="D")

    df = pd.DataFrame(
        {
            "data_atendimento": rng.choice(dates, rows),
            "regiao": rng.choice(regions, rows, p=[0.12, 0.2, 0.12, 0.34, 0.22]),
            "canal": rng.choice(channels, rows, p=[0.38, 0.28, 0.26, 0.08]),
            "produto": rng.choice(products, rows, p=[0.36, 0.28, 0.22, 0.14]),
            "idade_cliente": np.clip(rng.normal(39, 13, rows).round(), 18, 82),
            "tempo_espera_min": np.clip(rng.gamma(2.2, 8.5, rows), 1, 95).round(1),
            "valor_pago": np.clip(rng.normal(118, 32, rows), 35, 260).round(2),
            "satisfacao": np.clip(rng.normal(8.1, 1.2, rows), 1, 10).round(1),
            "campanha_ativa": rng.choice(["Sim", "Não"], rows, p=[0.45, 0.55]),
        }
    )

    # Outliers inseridos para que o usuário veja o critério de Tukey funcionando.
    outlier_rows = rng.choice(df.index, 8, replace=False)
    df.loc[outlier_rows[:4], "tempo_espera_min"] = rng.uniform(130, 210, 4).round(1)
    df.loc[outlier_rows[4:], "valor_pago"] = rng.uniform(310, 520, 4).round(2)

    # Valores ausentes e duplicidades simulam problemas comuns de bases reais.
    missing_wait = rng.choice(df.index, 14, replace=False)
    missing_sat = rng.choice(df.index, 10, replace=False)
    df.loc[missing_wait, "tempo_espera_min"] = np.nan
    df.loc[missing_sat, "satisfacao"] = np.nan
    duplicate_sample = df.sample(4, random_state=7)

    return pd.concat([df, duplicate_sample], ignore_index=True).sort_values("data_atendimento")


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Lê CSV enviado pelo usuário."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    raise ValueError("Formato não suportado. Use apenas arquivos CSV.")


def clean_column_names(columns: Iterable[str]) -> list[str]:
    """Padroniza nomes de colunas sem alterar seu significado."""
    cleaned = []
    seen: dict[str, int] = {}
    for col in columns:
        value = str(col).strip().replace("\n", " ").replace("\t", " ")
        value = " ".join(value.split())
        value = value if value else "coluna_sem_nome"
        base = value
        if base in seen:
            seen[base] += 1
            value = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
        cleaned.append(value)
    return cleaned


def prepare_dataframe(df: pd.DataFrame, normalize_names: bool) -> pd.DataFrame:
    """Aplica limpeza mínima: nomes de colunas e espaços extras em textos."""
    prepared = df.copy()
    if normalize_names:
        prepared.columns = clean_column_names(prepared.columns)
    for column in prepared.columns:
        if prepared[column].dtype == "object":
            prepared[column] = prepared[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    return prepared


# ============================================================
# 5) IDENTIFICAÇÃO DE TIPOS E SUMÁRIOS EXPLORATÓRIOS
# ============================================================
# Este bloco transforma a base em tabelas de leitura: estrutura, variáveis
# numéricas, variáveis categóricas, outliers, grupos e tempo.

def infer_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Detecta colunas temporais mesmo quando chegaram como texto."""
    datetime_columns: list[str] = []
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_columns.append(column)
            continue
        if not pd.api.types.is_object_dtype(df[column]):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().mean() >= 0.75:
            datetime_columns.append(column)
    return datetime_columns


def get_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Separa colunas numéricas, categóricas e temporais para alimentar a interface."""
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_columns = infer_datetime_columns(df)
    categorical_columns = [
        col
        for col in df.columns
        if col not in numeric_columns and col not in datetime_columns
    ]
    return numeric_columns, categorical_columns, datetime_columns


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume tipos, nulos e cardinalidade de cada coluna."""
    total_rows = len(df)
    result = pd.DataFrame(
        {
            "coluna": df.columns,
            "tipo": [str(df[col].dtype) for col in df.columns],
            "nulos": [int(df[col].isna().sum()) for col in df.columns],
            "percentual_nulos": [
                round(float(df[col].isna().mean() * 100), 2) for col in df.columns
            ],
            "valores_unicos": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )
    if total_rows == 0:
        result["percentual_nulos"] = 0.0
    return result.sort_values(["percentual_nulos", "nulos"], ascending=False)


def numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Calcula estatísticas robustas e clássicas para variáveis numéricas."""
    if not numeric_columns:
        return pd.DataFrame()

    rows = []
    for col in numeric_columns:
        series = df[col].dropna()
        if series.empty:
            rows.append(
                {
                    "variavel": col,
                    "n": 0,
                    "nulos": int(df[col].isna().sum()),
                    "media": np.nan,
                    "desvio_padrao": np.nan,
                    "minimo": np.nan,
                    "q1": np.nan,
                    "mediana": np.nan,
                    "q3": np.nan,
                    "maximo": np.nan,
                    "iqr": np.nan,
                }
            )
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        rows.append(
            {
                "variavel": col,
                "n": int(series.count()),
                "nulos": int(df[col].isna().sum()),
                "media": round(float(series.mean()), 3),
                "desvio_padrao": round(float(series.std()), 3),
                "minimo": round(float(series.min()), 3),
                "q1": round(float(q1), 3),
                "mediana": round(float(series.median()), 3),
                "q3": round(float(q3), 3),
                "maximo": round(float(series.max()), 3),
                "iqr": round(float(q3 - q1), 3),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    """Resume categorias, mostrando frequência e diversidade de valores."""
    rows = []
    for col in categorical_columns:
        counts = df[col].value_counts(dropna=True)
        top_value = counts.index[0] if not counts.empty else None
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        rows.append(
            {
                "variavel": col,
                "nulos": int(df[col].isna().sum()),
                "valores_unicos": int(df[col].nunique(dropna=True)),
                "categoria_mais_frequente": top_value,
                "frequencia": top_count,
                "percentual": round(top_count / len(df) * 100, 2) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Aplica o critério de Tukey: valores fora de Q1 - 1,5 IQR e Q3 + 1,5 IQR."""
    rows = []
    for col in numeric_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        rows.append(
            {
                "variavel": col,
                "limite_inferior": round(float(lower), 3),
                "limite_superior": round(float(upper), 3),
                "outliers": count,
                "percentual": round(float(count / len(series) * 100), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("outliers", ascending=False)


def build_group_summary(
    df: pd.DataFrame, numeric_column: str | None, group_column: str | None
) -> pd.DataFrame:
    """Compara a variável principal entre categorias escolhidas pelo usuário."""
    if not numeric_column or not group_column:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_column, dropna=False)[numeric_column]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    return grouped.round(3).sort_values("median", ascending=False)


def as_datetime_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Converte uma coluna para data quando necessário."""
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        return df[column]
    return pd.to_datetime(df[column], errors="coerce")


def build_time_summary(
    df: pd.DataFrame, numeric_column: str | None, time_column: str | None
) -> pd.DataFrame:
    """Agrega a variável principal por mês para enxergar tendência temporal."""
    if not numeric_column or not time_column:
        return pd.DataFrame()
    temp = df[[numeric_column, time_column]].copy()
    temp[time_column] = as_datetime_series(temp, time_column)
    temp = temp.dropna(subset=[time_column])
    if temp.empty:
        return pd.DataFrame()
    temp["periodo"] = temp[time_column].dt.to_period("M").dt.to_timestamp()
    return (
        temp.groupby("periodo")[numeric_column]
        .agg(["count", "mean", "median", "sum"])
        .reset_index()
        .round(3)
    )


# ============================================================
# 6) MODELAGEM EXPLICATIVA
# ============================================================
# A Random Forest é usada como aprofundamento opcional. Ela é adequada aqui
# porque lida bem com relações não lineares e permite estimar importância das
# variáveis. O app segue funcionando mesmo sem scikit-learn instalado.

def build_model_ready_data(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    """Prepara X e y, separando colunas por tipo e removendo alvo ausente."""
    model_df = df.dropna(subset=[target_column]).copy()
    y = model_df[target_column]
    X = model_df.drop(columns=[target_column])

    datetime_columns = infer_datetime_columns(X)
    for column in datetime_columns:
        parsed = as_datetime_series(X, column)
        X[f"{column}_ano"] = parsed.dt.year
        X[f"{column}_mes"] = parsed.dt.month
        X[f"{column}_dia"] = parsed.dt.day
        X = X.drop(columns=[column])

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        col
        for col in X.columns
        if col not in numeric_columns and col not in datetime_columns
    ]
    return X, y, numeric_columns, categorical_columns, datetime_columns


def feature_family(feature_name: str, original_columns: list[str]) -> str:
    """Agrupa dummies e atributos derivados de data de volta no campo original."""
    for column in sorted(original_columns, key=len, reverse=True):
        if feature_name == column or feature_name.startswith(f"{column}_"):
            return column
    return feature_name


def random_forest_influence(
    df: pd.DataFrame, target_column: str | None
) -> tuple[pd.DataFrame, dict[str, str | float | int]]:
    """Treina Random Forest e retorna importância agregada por campo da base."""
    metadata: dict[str, str | float | int] = {
        "status": "ok",
        "message": "",
        "metric_name": "R² de teste",
        "metric_value": np.nan,
        "rows_used": 0,
    }

    if target_column is None:
        metadata["status"] = "empty"
        metadata["message"] = "Selecione uma variável numérica principal para modelar."
        return pd.DataFrame(), metadata

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError:
        metadata["status"] = "missing_dependency"
        metadata["message"] = (
            "Para calcular Random Forest, instale a dependência com: "
            "python -m pip install scikit-learn"
        )
        return pd.DataFrame(), metadata

    X, y, numeric_columns, categorical_columns, datetime_columns = build_model_ready_data(
        df, target_column
    )
    valid_rows = int(y.notna().sum())
    metadata["rows_used"] = valid_rows

    if valid_rows < 30 or X.shape[1] == 0:
        metadata["status"] = "insufficient_data"
        metadata["message"] = (
            "A modelagem precisa de pelo menos 30 registros válidos e uma variável explicativa."
        )
        return pd.DataFrame(), metadata

    # OneHotEncoder mudou o nome do argumento sparse em versões recentes; este
    # fallback mantém compatibilidade com instalações diferentes do scikit-learn.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_depth=None,
        min_samples_leaf=3,
    )
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", model)])

    test_size = 0.25 if valid_rows >= 80 else 0.35
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    pipeline.fit(X_train, y_train)
    metric_value = float(pipeline.score(X_test, y_test))
    metadata["metric_value"] = round(metric_value, 3)

    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    original_columns = list(df.drop(columns=[target_column]).columns)

    rows = []
    for transformed_name, importance in zip(feature_names, importances):
        clean_name = transformed_name.split("__", 1)[-1]
        rows.append(
            {
                "campo": feature_family(clean_name, original_columns),
                "importancia": float(importance),
            }
        )

    result = (
        pd.DataFrame(rows)
        .groupby("campo", as_index=False)["importancia"]
        .sum()
        .sort_values("importancia", ascending=False)
    )
    total = result["importancia"].sum()
    if total > 0:
        result["importancia"] = result["importancia"] / total
    result["importancia_percentual"] = (result["importancia"] * 100).round(2)
    result = result.drop(columns=["importancia"]).head(12)

    if datetime_columns:
        metadata["message"] = (
            "Colunas de data foram transformadas em ano, mês e dia antes do modelo."
        )
    return result, metadata


# ============================================================
# 7) LEITURAS DIDÁTICAS E RECOMENDAÇÕES GERENCIAIS
# ============================================================
# Estas funções convertem números em frases de interpretação. Elas ajudam o
# usuário a enxergar implicações práticas sem confundir correlação com causa.

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Cria tabela Markdown sem depender do pacote opcional tabulate."""
    if df.empty:
        return ""

    formatted = df.astype(object).where(pd.notna(df), "")
    headers = [str(col) for col in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]

    header_line = "| " + " | ".join(
        headers[idx].ljust(widths[idx]) for idx in range(len(headers))
    ) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def explain_distribution(summary_row: pd.Series | None) -> str:
    """Explica centro, quartis e assimetria em linguagem simples."""
    if summary_row is None:
        return "Escolha uma variável numérica para gerar uma leitura da distribuição."

    median = summary_row["mediana"]
    mean = summary_row["media"]
    q1 = summary_row["q1"]
    q3 = summary_row["q3"]
    minimum = summary_row["minimo"]
    maximum = summary_row["maximo"]
    iqr = summary_row["iqr"]
    direction = "à direita" if mean > median else "à esquerda" if mean < median else "quase simétrica"
    management_hint = (
        "Para decisão gerencial, use a mediana como valor típico e o IQR como faixa de operação esperada. "
        "Valores fora dessa faixa pedem investigação antes de virar meta ou regra."
    )
    return (
        f"A mediana é {median}, então metade dos registros está abaixo desse valor. "
        f"O intervalo entre Q1 ({q1}) e Q3 ({q3}) concentra os 50% centrais da base; "
        f"esse intervalo tem amplitude {iqr}. Como a média ({mean}) e a mediana estão nessa relação, "
        f"a distribuição sugere assimetria {direction}. Os extremos observados vão de {minimum} a {maximum}. "
        f"{management_hint}"
    )


def explain_quality(missing: pd.DataFrame, duplicated_rows: int, total_rows: int) -> str:
    """Interpreta qualidade da base e sugere ação prática."""
    null_columns = missing[missing["nulos"] > 0] if not missing.empty else pd.DataFrame()
    parts = [
        f"A base tem {total_rows} linhas e {len(missing)} colunas.",
        f"Foram encontradas {duplicated_rows} linhas duplicadas.",
    ]
    if null_columns.empty:
        parts.append("Não há colunas com valores ausentes, o que reduz risco de viés por falta de informação.")
    else:
        worst = null_columns.iloc[0]
        parts.append(
            f"{len(null_columns)} coluna(s) têm valores ausentes. A mais crítica é {worst['coluna']}, "
            f"com {worst['percentual_nulos']}% de nulos."
        )
        parts.append(
            "Como caminho de atuação, priorize entender se os nulos são aleatórios ou ligados a algum grupo, canal, período ou processo."
        )
    if duplicated_rows:
        parts.append(
            "Antes de remover duplicidades, confirme se elas são registros repetidos por erro ou ocorrências reais com valores iguais."
        )
    return " ".join(parts)


def explain_outliers(outliers: pd.DataFrame) -> str:
    """Interpreta outliers e conecta com plano de ação."""
    if outliers.empty:
        return "Não foi possível calcular outliers por falta de variáveis numéricas."
    top = outliers.iloc[0]
    if top["outliers"] == 0:
        return (
            "Nenhuma variável numérica apresentou outliers pelo critério 1,5 x IQR. "
            "Mesmo assim, mantenha monitoramento se a variável for crítica para custo, prazo ou satisfação."
        )
    return (
        f"A variável com mais outliers é {top['variavel']}: {int(top['outliers'])} registro(s), "
        f"ou {top['percentual']}% dos valores válidos. Para gestão, trate esses casos como fila de investigação: "
        "audite cadastro, procure padrão por grupo ou data e decida se são erros, exceções ou casos especiais que exigem processo próprio."
    )


def explain_group_action(
    group_summary: pd.DataFrame, numeric_column: str | None, group_column: str | None
) -> str:
    """Explica diferença entre grupos com foco em decisão."""
    if group_summary.empty or numeric_column is None or group_column is None:
        return (
            "Nenhuma comparação entre grupos foi selecionada. Escolha uma variável categórica quando quiser saber "
            "onde agir primeiro: região, canal, produto, equipe, perfil de cliente ou outra segmentação."
        )
    ordered = group_summary.sort_values("median", ascending=False)
    leader = ordered.iloc[0]
    base = ordered.iloc[-1]
    gap = round(float(leader["median"] - base["median"]), 3)
    return (
        f"Na variável {numeric_column}, o grupo com maior mediana é {leader[group_column]} "
        f"e o menor é {base[group_column]}. A diferença entre medianas é {gap}. "
        "Como caminho gerencial, investigue práticas, demanda, equipe, mix de casos ou gargalos do grupo mais alto; "
        "se a variável for algo desejável, esse grupo pode ser referência. Se for algo indesejável, ele vira prioridade de ação."
    )


def explain_time_action(time_summary: pd.DataFrame, numeric_column: str | None) -> str:
    """Explica tendência temporal com implicação prática."""
    if time_summary.empty or numeric_column is None:
        return (
            "Nenhuma tendência temporal foi selecionada ou inferida. Use uma coluna de data quando quiser saber "
            "se o fenômeno está melhorando, piorando ou apenas variando pontualmente."
        )
    ordered = time_summary.sort_values("periodo")
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    direction = "aumentou" if last["mean"] > first["mean"] else "diminuiu" if last["mean"] < first["mean"] else "permaneceu estável"
    return (
        f"A média de {numeric_column} {direction} entre {first['periodo'].date()} e {last['periodo'].date()}. "
        "Para tomada de decisão, combine essa leitura com eventos do período: campanha, mudança de equipe, sazonalidade, "
        "alteração de preço, fila operacional ou mudança de regra."
    )


def explain_model_action(
    model_importance: pd.DataFrame,
    model_metadata: dict[str, str | float | int],
    target_column: str | None,
) -> str:
    """Explica a saída da Random Forest com ressalvas de interpretação."""
    if model_importance.empty:
        return str(model_metadata.get("message", "Modelo não calculado."))

    top = model_importance.iloc[0]
    metric = model_metadata.get("metric_value", np.nan)
    quality_note = (
        "Como o R² ficou abaixo de zero, o modelo não está prevendo bem fora da amostra; "
        "nesse caso, use a lista apenas como pista exploratória fraca e aprofunde a qualidade dos dados."
        if isinstance(metric, float) and metric < 0
        else "Como o modelo teve algum poder preditivo, os campos mais importantes são bons candidatos para investigação."
    )
    return (
        f"A Random Forest indicou que o campo mais influente para explicar {target_column} é {top['campo']}, "
        f"com importância relativa de {top['importancia_percentual']}%. O desempenho fora da amostra foi "
        f"R² = {metric}. {quality_note} Use isso como priorização investigativa, não como prova causal."
    )


def explain_r2(metric: str | float | int) -> str:
    """Explica R² em linguagem gerencial."""
    try:
        value = float(metric)
    except (TypeError, ValueError):
        return (
            "R² é uma medida de desempenho do modelo. Ele compara o modelo com uma previsão simples baseada na média."
        )

    if value < 0:
        interpretation = (
            "R² negativo significa que, no teste, o modelo foi pior do que simplesmente usar a média como previsão. "
            "Isso sinaliza que a base, as variáveis ou a relação entre elas ainda não sustentam uma boa previsão."
        )
    elif value < 0.3:
        interpretation = (
            "R² baixo indica que o modelo explica pouca parte da variação do indicador. A lista de campos influentes "
            "ainda pode gerar hipóteses, mas deve ser usada com cautela."
        )
    elif value < 0.7:
        interpretation = (
            "R² intermediário indica que o modelo capturou parte relevante do padrão, mas ainda há fatores não explicados."
        )
    else:
        interpretation = (
            "R² alto indica que o modelo conseguiu explicar boa parte da variação do indicador nos dados de teste."
        )
    return (
        "R², chamado de coeficiente de determinação, mostra o quanto o modelo consegue explicar da variação "
        f"da variável analisada em dados separados para teste. {interpretation}"
    )


def render_numeric_glossary() -> None:
    """Mostra uma explicação curta para cada medida do resumo numérico."""
    st.markdown("#### Como ler esta tabela")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.write("- **n**: quantidade de registros válidos usados no cálculo.")
        st.write("- **nulos**: registros sem informação naquela variável.")
        st.write("- **média**: soma dos valores dividida pela quantidade de registros.")
        st.write("- **desvio padrão**: mede o quanto os valores se espalham em torno da média.")
        st.write("- **mínimo e máximo**: menor e maior valor observados.")
    with col2:
        st.write("- **mediana**: valor central dos dados ordenados; metade fica abaixo e metade acima.")
        st.write("- **Q1**: primeiro quartil; 25% dos valores ficam abaixo dele.")
        st.write("- **Q3**: terceiro quartil; 75% dos valores ficam abaixo dele.")
        st.write("- **IQR**: Q3 menos Q1; faixa dos 50% centrais da base.")
        st.write("- **outlier**: valor muito distante do padrão central, que merece investigação.")


def default_numeric_index(numeric_columns: list[str]) -> int | None:
    """Escolhe um indicador inicial mais útil quando a base exemplo está em uso."""
    if not numeric_columns:
        return None
    preferred_terms = ["tempo", "valor", "satisfacao", "satisfação", "custo", "nota"]
    for term in preferred_terms:
        for idx, column in enumerate(numeric_columns):
            if term in column.lower():
                return idx
    return 0


def build_report_markdown(
    ctx: EDAContext,
    missing: pd.DataFrame,
    numeric: pd.DataFrame,
    outliers: pd.DataFrame,
    group_summary: pd.DataFrame,
    time_summary: pd.DataFrame,
    model_importance: pd.DataFrame,
    model_metadata: dict[str, str | float | int],
) -> str:
    """Monta o relatório exportável em Markdown."""
    rows, cols = ctx.data.shape
    lines = [
        "# Relatório EDA orientado por Tukey",
        "",
        f"Pergunta de análise: {ctx.question or 'não informada'}",
        f"Unidade de análise: {ctx.unit or 'não informada'}",
        f"Dimensão da base: {rows} linhas e {cols} colunas",
        "",
        "## Qualidade e estrutura",
        explain_quality(missing, int(ctx.data.duplicated().sum()), len(ctx.data)),
        "",
        "## Variáveis numéricas",
        dataframe_to_markdown(numeric) if not numeric.empty else "Nenhuma variável numérica encontrada.",
        "",
        "## Leitura da variável principal",
        explain_distribution(
            numeric[numeric["variavel"] == ctx.selected_numeric].iloc[0]
            if ctx.selected_numeric and not numeric[numeric["variavel"] == ctx.selected_numeric].empty
            else None
        ),
        "",
        "## Outliers pelo critério de Tukey",
        dataframe_to_markdown(outliers) if not outliers.empty else "Nenhum resumo de outliers disponível.",
        explain_outliers(outliers),
        "",
    ]
    if not group_summary.empty:
        lines.extend(
            [
                "## Comparação entre grupos",
                dataframe_to_markdown(group_summary),
                explain_group_action(group_summary, ctx.selected_numeric, ctx.selected_group),
                "",
            ]
        )
    if not time_summary.empty:
        lines.extend(
            [
                "## Tendência temporal",
                dataframe_to_markdown(time_summary),
                explain_time_action(time_summary, ctx.selected_numeric),
                "",
            ]
        )
    lines.extend(
        [
            "## Influência dos campos por Random Forest",
            dataframe_to_markdown(model_importance)
            if not model_importance.empty
            else str(model_metadata.get("message", "Modelo não calculado.")),
            explain_model_action(model_importance, model_metadata, ctx.selected_numeric),
            explain_r2(model_metadata.get("metric_value", np.nan)),
            "",
            "## Próximos passos",
            "Validar registros ausentes, revisar outliers antes de removê-los, priorizar grupos ou campos influentes e aprofundar com testes estatísticos, entrevistas de processo ou experimentos controlados quando necessário.",
        ]
    )
    return "\n".join(lines)


def truncate_pdf_text(value: object, limit: int = 42) -> str:
    """Encurta textos longos para evitar que tabelas do PDF estourem a página."""
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def dataframe_to_pdf_table(df: pd.DataFrame, max_rows: int = 18) -> list[list[str]]:
    """Converte DataFrame em matriz de texto para tabela ReportLab."""
    if df.empty:
        return []
    limited = df.head(max_rows).copy()
    table = [[truncate_pdf_text(col, 30) for col in limited.columns]]
    for row in limited.to_numpy():
        table.append([truncate_pdf_text(value) for value in row])
    if len(df) > max_rows:
        table.append([f"... mais {len(df) - max_rows} linha(s)" ] + [""] * (len(table[0]) - 1))
    return table


def build_report_pdf(
    ctx: EDAContext,
    missing: pd.DataFrame,
    numeric: pd.DataFrame,
    outliers: pd.DataFrame,
    group_summary: pd.DataFrame,
    time_summary: pd.DataFrame,
    model_importance: pd.DataFrame,
    model_metadata: dict[str, str | float | int],
) -> bytes:
    """Gera o relatório em PDF com textos didáticos e tabelas principais."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.1 * cm,
        bottomMargin=1.1 * cm,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="JustifiedBody",
            parent=styles["BodyText"],
            alignment=TA_JUSTIFY,
            fontSize=9.5,
            leading=13,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#4b5563"),
        )
    )

    def add_title(story: list, text: str) -> None:
        story.append(Paragraph(text, styles["Title"]))
        story.append(Spacer(1, 0.18 * cm))

    def add_heading(story: list, text: str) -> None:
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(text, styles["Heading2"]))

    def add_paragraph(story: list, text: str, style_name: str = "JustifiedBody") -> None:
        story.append(Paragraph(str(text).replace("\n", "<br/>"), styles[style_name]))

    def add_table(story: list, df: pd.DataFrame) -> None:
        table_data = dataframe_to_pdf_table(df)
        if not table_data:
            add_paragraph(story, "Nenhuma tabela disponível.")
            return
        col_width = doc.width / max(len(table_data[0]), 1)
        table = Table(table_data, repeatRows=1, colWidths=[col_width] * len(table_data[0]))
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("LEADING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.2 * cm))

    selected_summary = None
    if ctx.selected_numeric and not numeric.empty:
        matches = numeric[numeric["variavel"] == ctx.selected_numeric]
        if not matches.empty:
            selected_summary = matches.iloc[0]

    story: list = []
    add_title(story, "Relatório EDA orientado pelos passos de Tukey")
    add_paragraph(story, f"Pergunta de análise: {ctx.question or 'não informada'}")
    add_paragraph(story, f"Unidade de análise: {ctx.unit or 'não informada'}")
    add_paragraph(story, f"Dimensão da base: {len(ctx.data)} linhas e {ctx.data.shape[1]} colunas")
    add_paragraph(story, f"EDA: {EDA_DEFINITION}", "SmallNote")
    add_paragraph(story, f"Tukey: {TUKEY_DEFINITION}", "SmallNote")

    add_heading(story, "1. Qualidade e estrutura")
    add_paragraph(story, explain_quality(missing, int(ctx.data.duplicated().sum()), len(ctx.data)))
    add_table(story, missing)

    add_heading(story, "2. Variáveis numéricas")
    add_paragraph(story, TUKEY_EXPLANATIONS["numeric"])
    add_table(story, numeric)

    add_heading(story, "3. Leitura da variável principal")
    add_paragraph(story, explain_distribution(selected_summary))

    add_heading(story, "4. Outliers")
    add_paragraph(story, TUKEY_EXPLANATIONS["outlier"])
    add_paragraph(story, explain_outliers(outliers))
    add_table(story, outliers)

    if not group_summary.empty:
        add_heading(story, "5. Comparação entre grupos")
        add_paragraph(story, explain_group_action(group_summary, ctx.selected_numeric, ctx.selected_group))
        add_table(story, group_summary)

    if not time_summary.empty:
        add_heading(story, "6. Tendência temporal")
        add_paragraph(story, explain_time_action(time_summary, ctx.selected_numeric))
        add_table(story, time_summary)

    add_heading(story, "7. Influência dos campos por Random Forest")
    add_paragraph(story, TUKEY_EXPLANATIONS["model"])
    add_paragraph(story, explain_model_action(model_importance, model_metadata, ctx.selected_numeric))
    add_paragraph(story, explain_r2(model_metadata.get("metric_value", np.nan)))
    add_table(story, model_importance)

    add_heading(story, "8. Próximos passos")
    add_paragraph(
        story,
        "Validar registros ausentes, revisar outliers antes de removê-los, priorizar grupos ou campos "
        "influentes e aprofundar com testes estatísticos, entrevistas de processo ou experimentos controlados.",
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# 8) GRÁFICOS
# ============================================================
# Os gráficos são mantidos em funções separadas para que a parte visual da EDA
# fique fácil de localizar e alterar.

def histogram_chart(df: pd.DataFrame, column: str) -> alt.Chart:
    """Histograma da variável principal."""
    return (
        alt.Chart(df[[column]].dropna())
        .mark_bar(color="#1d4ed8")
        .encode(
            alt.X(f"{column}:Q", bin=alt.Bin(maxbins=30), title=column),
            alt.Y("count():Q", title="Frequência"),
            tooltip=[alt.Tooltip("count():Q", title="Registros")],
        )
        .properties(height=260)
    )


def boxplot_chart(df: pd.DataFrame, column: str, group_column: str | None = None) -> alt.Chart:
    """Boxplot geral ou por grupo."""
    data = df[[column] + ([group_column] if group_column else [])].dropna()
    if group_column:
        return (
            alt.Chart(data)
            .mark_boxplot(extent=1.5, color="#1d4ed8")
            .encode(
                alt.X(f"{group_column}:N", title=group_column, sort="-y"),
                alt.Y(f"{column}:Q", title=column),
            )
            .properties(height=320)
        )
    return (
        alt.Chart(data)
        .mark_boxplot(extent=1.5, color="#1d4ed8")
        .encode(alt.Y(f"{column}:Q", title=column))
        .properties(height=260)
    )


def time_chart(time_summary: pd.DataFrame, numeric_column: str) -> alt.Chart:
    """Linha temporal mensal da média da variável principal."""
    return (
        alt.Chart(time_summary)
        .mark_line(point=True, color="#2563eb")
        .encode(
            alt.X("periodo:T", title="Período"),
            alt.Y("mean:Q", title=f"Média de {numeric_column}"),
            tooltip=[
                alt.Tooltip("periodo:T", title="Período"),
                alt.Tooltip("count:Q", title="Registros"),
                alt.Tooltip("mean:Q", title="Média"),
                alt.Tooltip("median:Q", title="Mediana"),
            ],
        )
        .properties(height=300)
    )


def model_importance_chart(model_importance: pd.DataFrame) -> alt.Chart:
    """Gráfico de barras da importância relativa dos campos."""
    return (
        alt.Chart(model_importance)
        .mark_bar(color="#334155")
        .encode(
            alt.X("importancia_percentual:Q", title="Importância relativa (%)"),
            alt.Y("campo:N", title="Campo", sort="-x"),
            tooltip=[
                alt.Tooltip("campo:N", title="Campo"),
                alt.Tooltip("importancia_percentual:Q", title="Importância (%)"),
            ],
        )
        .properties(height=320)
    )


# ============================================================
# 9) COMPONENTES DA INTERFACE
# ============================================================
# Este bloco desenha a experiência do usuário: introdução, menu lateral, carga,
# abas de EDA e tela final de resultados.

def render_intro() -> None:
    """Mostra a apresentação inicial do app."""
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.title("Análise Exploratória de Dados (com base nos passos de Tukey*)")
    st.markdown(
        f"""
        <div class="intro">
            <strong>O que é EDA?</strong> {EDA_DEFINITION}
            <br><br>
            Este app conduz a análise do começo ao fim: primeiro entende a pergunta e a unidade de análise,
            depois verifica a estrutura da base, calcula medidas como média, mediana e quartis,
            visualiza distribuições, identifica outliers, compara grupos, observa tendência temporal
            e transforma os achados em hipóteses, decisões e planos de ação.
            <br><br>
            <span class="small-muted"><strong>*Quem é Tukey?</strong> {TUKEY_DEFINITION}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame) -> tuple[str, str, str | None, str | None, str | None]:
    """Renderiza o menu lateral e captura as escolhas do usuário."""
    numeric_columns, categorical_columns, datetime_columns = get_column_groups(df)

    st.sidebar.header("Configuração da análise")
    st.sidebar.caption(
        "Use este painel para dizer qual pergunta será analisada e quais campos devem orientar a leitura."
    )

    st.sidebar.header("Pergunta")
    question = st.sidebar.text_area(
        "Qual pergunta a EDA deve iluminar?",
        value="Quais padrões explicam diferenças de tempo de espera e satisfação?",
        height=90,
        help=(
            "Escreva a dúvida gerencial ou analítica que orienta a leitura. Exemplo: "
            "'Quais fatores aumentam o tempo de espera?'"
        ),
    )
    st.sidebar.caption(
        "A pergunta evita que a análise vire uma lista de gráficos. Ela define o que é relevante observar."
    )

    unit = st.sidebar.text_input(
        "Unidade de análise",
        value="Cada linha representa um atendimento",
        help=(
            "Explique o que cada linha da base representa: um cliente, venda, atendimento, mês, loja, contrato etc."
        ),
    )
    st.sidebar.caption(
        "A unidade de análise impede comparações erradas. Uma linha por cliente não deve ser lida como uma linha por venda."
    )

    st.sidebar.header("Foco da leitura")
    selected_numeric = st.sidebar.selectbox(
        "Variável numérica principal",
        options=numeric_columns,
        index=default_numeric_index(numeric_columns),
        placeholder="Nenhuma variável numérica",
        help=(
            "Escolha o campo que você quer explicar, acompanhar ou melhorar. Ele precisa ser numérico, "
            "como custo, tempo, valor, nota, idade, produção ou volume."
        ),
    )
    st.sidebar.caption(
        "É o indicador central da EDA. Os gráficos, comparações e Random Forest tentarão explicar esse campo."
    )

    selected_group = st.sidebar.selectbox(
        "Variável para comparar grupos",
        options=[None] + categorical_columns,
        format_func=lambda value: "Sem comparação" if value is None else value,
        help=(
            "Escolha um campo de categorias para comparar o indicador principal entre grupos, "
            "como região, canal, produto, equipe ou status."
        ),
    )
    st.sidebar.caption(
        "Use para descobrir onde há diferenças práticas. Isso ajuda a priorizar ações por segmento."
    )

    selected_time = st.sidebar.selectbox(
        "Variável temporal",
        options=[None] + datetime_columns,
        format_func=lambda value: "Sem tendência temporal" if value is None else value,
        help=(
            "Escolha uma coluna de data para observar se o indicador muda ao longo do tempo."
        ),
    )
    st.sidebar.caption(
        "Use quando a pergunta envolve evolução, sazonalidade, piora, melhora ou efeito de mudanças ao longo dos meses."
    )

    return question, unit, selected_numeric, selected_group, selected_time


def render_load_screen() -> pd.DataFrame:
    """Tela 1: carrega arquivo ou usa base exemplo."""
    st.subheader("1. Carga da base de dados")
    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        uploaded_file = st.file_uploader(
            "Carregue uma base CSV",
            type=["csv"],
            help="Use um arquivo .csv. O app tenta detectar automaticamente o separador do arquivo.",
        )
        normalize_names = st.checkbox(
            "Padronizar nomes das colunas e remover espaços extras em textos",
            value=True,
            help="Remove espaços acidentais e evita colunas duplicadas com o mesmo nome visual.",
        )
    with right:
        st.markdown(
            f"""
            <div class="eda-note">
                {TUKEY_EXPLANATIONS["estrutura"]}
                Nesta etapa, a principal decisão é confirmar se a base representa bem o fenômeno que será analisado.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is None:
        st.info("Nenhum arquivo carregado. Usando uma base exemplo de atendimentos.")
        df = build_example_dataset()
    else:
        try:
            df = read_uploaded_file(uploaded_file)
            st.success(f"Base carregada: {uploaded_file.name}")
        except Exception as exc:
            st.error(f"Não foi possível carregar o arquivo: {exc}")
            st.stop()

    df = prepare_dataframe(df, normalize_names=normalize_names)
    st.dataframe(df.head(30), use_container_width=True)
    return df


def render_eda_screen(
    ctx: EDAContext,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tela 2: mostra as análises exploratórias em abas."""
    df = ctx.data
    st.subheader("2. Análise Exploratória de Dados")
    st.markdown(
        f"""
        <div class="intro">
            {EDA_DEFINITION}
            Nesta seção, cada aba responde uma pergunta prática: a base é confiável?
            Como os números se comportam? Há valores extremos? Existem diferenças entre grupos?
            O indicador muda ao longo do tempo?
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing = missing_summary(df)
    numeric = numeric_summary(df, ctx.numeric_columns)
    categorical = categorical_summary(df, ctx.categorical_columns)
    outliers = outlier_summary(df, ctx.numeric_columns)
    group_summary = build_group_summary(df, ctx.selected_numeric, ctx.selected_group)
    time_summary = build_time_summary(df, ctx.selected_numeric, ctx.selected_time)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linhas", f"{len(df):,}".replace(",", "."))
    m2.metric("Colunas", f"{df.shape[1]}")
    m3.metric("Duplicadas", f"{int(df.duplicated().sum())}")
    m4.metric("Colunas com nulos", f"{int((missing['nulos'] > 0).sum())}")

    structure_tab, numeric_tab, visual_tab, group_tab, time_tab = st.tabs(
        ["Estrutura", "Resumo numérico", "Distribuições", "Grupos", "Tempo"]
    )

    with structure_tab:
        st.markdown("#### Estrutura, tipos e qualidade")
        st.markdown(f'<div class="eda-note">{TUKEY_EXPLANATIONS["estrutura"]}</div>', unsafe_allow_html=True)
        st.dataframe(missing, use_container_width=True)
        st.write(explain_quality(missing, int(df.duplicated().sum()), len(df)))
        if int(df.duplicated().sum()) > 0:
            st.markdown(
                """
                <div class="eda-warning">
                    Há linhas duplicadas. Em EDA, isso não deve ser removido automaticamente:
                    primeiro confirme se são repetições indevidas ou registros reais com valores iguais.
                </div>
                """,
                unsafe_allow_html=True,
            )
        if not categorical.empty:
            st.markdown("#### Variáveis categóricas")
            st.caption(
                "Categorias mostram como a base se divide. Valores únicos demais podem indicar identificadores, "
                "não grupos úteis para comparação."
            )
            st.dataframe(categorical, use_container_width=True)

    with numeric_tab:
        st.markdown(f'<div class="eda-note">{TUKEY_EXPLANATIONS["numeric"]}</div>', unsafe_allow_html=True)
        if numeric.empty:
            st.warning("Nenhuma variável numérica foi encontrada.")
        else:
            st.markdown("#### Mínimo, máximo, mediana, quartis e dispersão")
            st.dataframe(numeric, use_container_width=True)
            render_numeric_glossary()

        if outliers.empty:
            st.info("Nenhum resumo de outliers disponível.")
        else:
            st.markdown("#### Outliers pelo critério de Tukey")
            st.markdown(f'<div class="eda-note">{TUKEY_EXPLANATIONS["outlier"]}</div>', unsafe_allow_html=True)
            st.dataframe(outliers, use_container_width=True)
            st.write(explain_outliers(outliers))

    with visual_tab:
        st.markdown(f'<div class="eda-note">{TUKEY_EXPLANATIONS["visual"]}</div>', unsafe_allow_html=True)
        if ctx.selected_numeric is None:
            st.warning("Selecione uma variável numérica na barra lateral.")
        else:
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("#### Histograma")
                st.caption(
                    "Mostra concentração, caudas e assimetria. Picos podem indicar perfis diferentes dentro da mesma base."
                )
                st.altair_chart(histogram_chart(df, ctx.selected_numeric), use_container_width=True)
            with right:
                st.markdown("#### Boxplot")
                st.caption(
                    "Mostra mediana, quartis e valores extremos. É uma das ferramentas clássicas de Tukey."
                )
                st.altair_chart(boxplot_chart(df, ctx.selected_numeric), use_container_width=True)

    with group_tab:
        st.markdown(f'<div class="eda-note">{TUKEY_EXPLANATIONS["groups"]}</div>', unsafe_allow_html=True)
        if ctx.selected_numeric is None or ctx.selected_group is None:
            st.info("Escolha uma variável numérica e uma variável de grupo para comparar.")
        else:
            st.markdown("#### Comparação entre grupos")
            st.dataframe(group_summary, use_container_width=True)
            st.write(explain_group_action(group_summary, ctx.selected_numeric, ctx.selected_group))
            st.altair_chart(
                boxplot_chart(df, ctx.selected_numeric, ctx.selected_group),
                use_container_width=True,
            )

    with time_tab:
        st.markdown(f'<div class="eda-note">{TUKEY_EXPLANATIONS["time"]}</div>', unsafe_allow_html=True)
        if ctx.selected_numeric is None or ctx.selected_time is None:
            st.info("Escolha uma variável numérica e uma variável temporal para analisar tendência.")
        elif time_summary.empty:
            st.warning("Não foi possível montar a série temporal com as escolhas atuais.")
        else:
            st.markdown("#### Tendência temporal mensal")
            st.dataframe(time_summary, use_container_width=True)
            st.write(explain_time_action(time_summary, ctx.selected_numeric))
            st.altair_chart(time_chart(time_summary, ctx.selected_numeric), use_container_width=True)

    return missing, numeric, outliers, group_summary, time_summary


def render_results_screen(
    ctx: EDAContext,
    missing: pd.DataFrame,
    numeric: pd.DataFrame,
    outliers: pd.DataFrame,
    group_summary: pd.DataFrame,
    time_summary: pd.DataFrame,
    model_importance: pd.DataFrame,
    model_metadata: dict[str, str | float | int],
) -> None:
    """Tela 3: transforma as análises em interpretação e relatório."""
    st.subheader("3. Resultados da EDA explicados de forma didática")
    st.markdown(
        """
        <div class="intro">
            Esta tela transforma os resultados em leitura analítica. A ideia é sair de tabelas soltas
            para uma narrativa: o que foi observado, por que importa, qual decisão pode apoiar
            e o que investigar em seguida.
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_summary = None
    if ctx.selected_numeric and not numeric.empty:
        matches = numeric[numeric["variavel"] == ctx.selected_numeric]
        if not matches.empty:
            selected_summary = matches.iloc[0]

    with st.expander("1. Pergunta e unidade de análise", expanded=True):
        st.write(f"Pergunta orientadora: {ctx.question or 'não informada'}")
        st.write(f"Unidade de análise: {ctx.unit or 'não informada'}")
        st.caption(
            "Sem pergunta e unidade de análise, a EDA vira uma coleção de gráficos. Com elas, cada resultado ganha contexto."
        )

    with st.expander("2. Estrutura e qualidade da base", expanded=True):
        st.write(explain_quality(missing, int(ctx.data.duplicated().sum()), len(ctx.data)))
        st.caption(
            "Valores ausentes e duplicidades afetam médias, proporções, comparações entre grupos e conclusões posteriores."
        )

    with st.expander("3. Centro, dispersão e forma da distribuição", expanded=True):
        st.write(explain_distribution(selected_summary))
        st.caption(
            "No espírito de Tukey, a mediana e os quartis ajudam a descrever os dados antes de impor qualquer modelo."
        )

    with st.expander("4. Outliers", expanded=True):
        st.write(explain_outliers(outliers))
        st.caption(
            "Outlier não é sinônimo de erro. Pode ser falha de registro, caso raro ou sinal importante do fenômeno."
        )

    with st.expander("5. Comparação entre grupos", expanded=True):
        st.write(explain_group_action(group_summary, ctx.selected_numeric, ctx.selected_group))
        st.caption(
            "Compare medianas e amplitudes antes de comparar médias: grupos com muitos extremos podem distorcer a leitura."
        )

    with st.expander("6. Tendência temporal", expanded=True):
        st.write(explain_time_action(time_summary, ctx.selected_numeric))
        st.caption(
            "Tendência temporal é uma pista: depois dela, vale investigar sazonalidade, mudanças operacionais e eventos externos."
        )

    with st.expander("7. Influência dos campos por Random Forest", expanded=True):
        st.markdown(f'<div class="eda-note">{TUKEY_EXPLANATIONS["model"]}</div>', unsafe_allow_html=True)
        st.write(
            "Na prática, pense na Random Forest como várias perguntas em sequência feitas aos dados. "
            "Uma árvore pode perguntar, por exemplo, se o canal é aplicativo ou loja; outra pode olhar valor, data ou região. "
            "O conjunto dessas árvores aprende padrões e estima quais campos mais ajudaram nas previsões."
        )
        if model_importance.empty:
            st.warning(str(model_metadata.get("message", "Modelo não calculado.")))
        else:
            col_a, col_b = st.columns([0.48, 0.52], gap="large")
            with col_a:
                st.dataframe(model_importance, use_container_width=True)
                st.metric(
                    str(model_metadata.get("metric_name", "R² de teste")),
                    str(model_metadata.get("metric_value", "n/a")),
                )
            with col_b:
                st.altair_chart(model_importance_chart(model_importance), use_container_width=True)
            st.write(explain_model_action(model_importance, model_metadata, ctx.selected_numeric))
            st.write(explain_r2(model_metadata.get("metric_value", np.nan)))
            if model_metadata.get("message"):
                st.caption(str(model_metadata["message"]))
            st.caption(
                "Importância de variável mostra contribuição preditiva no modelo. Ela não prova causalidade; serve para priorizar investigação, entrevistas de processo e testes posteriores."
            )

    with st.expander("8. Hipóteses e próximos passos", expanded=True):
        hypotheses = []
        if not group_summary.empty:
            hypotheses.append(
                f"Investigar por que {ctx.selected_group} altera a distribuição de {ctx.selected_numeric}."
            )
        if not outliers.empty and outliers.iloc[0]["outliers"] > 0:
            hypotheses.append(
                f"Auditar registros extremos em {outliers.iloc[0]['variavel']} para separar erro, exceção e oportunidade."
            )
        if not time_summary.empty:
            hypotheses.append(
                f"Verificar se mudanças no tempo explicam variações em {ctx.selected_numeric}."
            )
        if not model_importance.empty:
            top_field = model_importance.iloc[0]["campo"]
            hypotheses.append(
                f"Priorizar uma análise de plano de ação envolvendo {top_field}, campo mais influente no modelo."
            )
        if not hypotheses:
            hypotheses.append(
                "Ampliar a base ou definir uma segmentação mais informativa para gerar hipóteses testáveis."
            )
        for item in hypotheses:
            st.write(f"- {item}")

    try:
        report_pdf = build_report_pdf(
            ctx,
            missing,
            numeric,
            outliers,
            group_summary,
            time_summary,
            model_importance,
            model_metadata,
        )
        st.download_button(
            "Baixar relatório em PDF",
            data=report_pdf,
            file_name="relatorio_eda_tukey.pdf",
            mime="application/pdf",
        )
    except ImportError:
        st.warning(
            "Para baixar o relatório em PDF, instale a dependência com: python -m pip install reportlab"
        )


# ============================================================
# 10) ORQUESTRAÇÃO DO APP
# ============================================================
# A função main organiza a sequência pedida: carga da base, EDA e tela final
# de interpretação. O Streamlit reexecuta este fluxo a cada interação.

def main() -> None:
    """Executa o fluxo completo da aplicação."""
    render_intro()
    df = render_load_screen()
    numeric_columns, categorical_columns, datetime_columns = get_column_groups(df)
    question, unit, selected_numeric, selected_group, selected_time = render_sidebar(df)

    ctx = EDAContext(
        data=df,
        question=question,
        unit=unit,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        selected_numeric=selected_numeric,
        selected_group=selected_group,
        selected_time=selected_time,
    )

    missing, numeric, outliers, group_summary, time_summary = render_eda_screen(ctx)
    model_importance, model_metadata = random_forest_influence(df, selected_numeric)
    render_results_screen(
        ctx,
        missing,
        numeric,
        outliers,
        group_summary,
        time_summary,
        model_importance,
        model_metadata,
    )


if __name__ == "__main__":
    main()
