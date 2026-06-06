import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

# ==================================================
# CONFIGURACIÓN DE PÁGINA
# ==================================================

st.set_page_config(
    page_title="Dashboard Bibliométrico",
    page_icon="🍎",
    layout="wide"
)

# ==================================================
# CARGA DE DATOS
# ==================================================

@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/RicardoCP18/machine-learning-fruits-dashboard/main/data/scopus_PA3.csv"
    return pd.read_csv(url)

df = cargar_datos()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🍎 Dashboard Bibliométrico")

st.sidebar.markdown("""
### Keywords

- Machine Learning
- Fruits
- Classification
- Nutrition
""")

st.sidebar.markdown("""
### Fuente de Datos

Scopus
""")

st.sidebar.markdown("""
### Autor

Ricardo Céspedes
""")

# ==================================================
# ENCABEZADO
# ==================================================

st.title("🍎 Dashboard Bibliométrico sobre Machine Learning y Fruits")

st.markdown("""
Bienvenido al dashboard de análisis bibliométrico.

Esta aplicación permite explorar publicaciones científicas obtenidas desde Scopus relacionadas con Machine Learning, clasificación de frutas y nutrición.

A través de visualizaciones interactivas se pueden identificar tendencias de investigación, autores destacados, artículos influyentes y conceptos frecuentes presentes en la literatura científica.
""")

# ==================================================
# PREGUNTA DE INVESTIGACIÓN
# ==================================================

st.subheader("Pregunta de Investigación")

st.info("""
¿Cómo se aplica el Machine Learning en la clasificación de frutas utilizando información nutricional para apoyar la toma de decisiones alimentarias saludables?
""")

# ==================================================
# TABS
# ==================================================

# ==================================================
# FILTRO DE AÑO
# ==================================================

st.subheader("Filtro de Información")

col1, col2 = st.columns([1, 3])

with col1:

    opciones_anio = ["Todos"] + sorted(
        df["Year"].dropna().unique().tolist(),
        reverse=True
    )

    anio_seleccionado = st.selectbox(
        "Seleccione un año:",
        opciones_anio
    )

if anio_seleccionado == "Todos":
    df_filtrado = df.copy()
else:
    df_filtrado = df[df["Year"] == anio_seleccionado]

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"Mostrando {len(df_filtrado)} registros para: {anio_seleccionado}"
    )

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3 = st.tabs(
    ["📊 Resumen", "📈 Dashboard", "📄 Dataset"]
)

# ==================================================
# TAB 1 - RESUMEN
# ==================================================

with tab1:

    st.subheader("Resumen General")

    autores_unicos = set()

    for fila in df_filtrado["Authors"].dropna():
        for autor in fila.split(";"):
            autores_unicos.add(autor.strip())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Artículos",
        len(df_filtrado)
    )

    col2.metric(
        "Autores únicos",
        len(autores_unicos)
    )

    col3.metric(
        "Años analizados",
        df_filtrado["Year"].nunique()
    )

    col4.metric(
        "Total citas",
        int(df_filtrado["Cited by"].fillna(0).sum())
    )

    st.subheader("Conclusiones")

    st.success("""
    La producción científica relacionada con Machine Learning aplicado a frutas y nutrición muestra una tendencia de crecimiento.

    Los artículos más citados evidencian el interés de la comunidad científica por técnicas de clasificación y análisis de datos.

    Los conceptos frecuentes encontrados en los abstracts reflejan una estrecha relación entre aprendizaje automático, clasificación y nutrición.
    """)

# ==================================================
# TAB 2 - DASHBOARD
# ==================================================

with tab2:

    # ----------------------------------------------
    # GRÁFICO 1
    # ----------------------------------------------

    st.subheader("📈 Publicaciones por Año")

    st.info("""
    Este gráfico permite observar la evolución temporal de las publicaciones científicas relacionadas con la temática estudiada.
    """)

    publicaciones = df["Year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        publicaciones.index,
        publicaciones.values
    )

    ax.set_xlabel("Año")
    ax.set_ylabel("Cantidad de publicaciones")

    st.pyplot(fig)

    # ----------------------------------------------
    # GRÁFICO 2
    # ----------------------------------------------

    st.subheader("🏆 Top 10 Artículos Más Citados")

    st.info("""
    Permite identificar los artículos con mayor impacto académico dentro del conjunto analizado.
    """)

    top_citados = (
        df.sort_values(
            by="Cited by",
            ascending=False
        )
        .head(10)
        .copy()
    )

    top_citados["Titulo Corto"] = (
        top_citados["Title"]
        .str[:50]
        + "..."
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(
        top_citados["Titulo Corto"],
        top_citados["Cited by"]
    )

    ax.set_xlabel("Número de citas")

    plt.subplots_adjust(left=0.35)

    st.pyplot(fig)

    st.subheader("Detalle de los Artículos Más Citados")

    st.dataframe(
        top_citados[
            ["Title", "Authors", "Year", "Cited by"]
        ]
    )

    # ----------------------------------------------
    # GRÁFICO 3
    # ----------------------------------------------

    st.subheader("👨‍🔬 Autores con Más Publicaciones")

    st.info("""
    Muestra los investigadores con mayor participación dentro del conjunto de publicaciones analizadas.
    """)

    autores = []

    for fila in df["Authors"].dropna():
        autores.extend(
            [a.strip() for a in fila.split(";")]
        )

    contador = Counter(autores)

    top_autores = contador.most_common(10)

    autores_df = pd.DataFrame(
        top_autores,
        columns=["Autor", "Publicaciones"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        autores_df["Autor"],
        autores_df["Publicaciones"]
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # ----------------------------------------------
    # GRÁFICO 4
    # ----------------------------------------------

    st.subheader("📚 Revistas con Más Publicaciones")

    st.info("""
    Identifica las fuentes científicas que publican más investigaciones sobre la temática.
    """)

    top_revistas = (
        df["Source title"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    top_revistas.sort_values().plot(
    kind="barh",
    ax=ax
    )

    ax.set_xlabel("Cantidad de publicaciones")
    ax.set_ylabel("Revista")
    
    plt.tight_layout()

    st.pyplot(fig)

    # ----------------------------------------------
    # GRÁFICO 5
    # ----------------------------------------------

    st.subheader("📄 Tipos de Documentos")

    st.info("""
    Permite conocer la distribución de artículos, revisiones y otros tipos de documentos científicos.
    """)

    tipos = df["Document Type"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        tipos,
        labels=tipos.index,
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    # ----------------------------------------------
    # GRÁFICO 6
    # ----------------------------------------------

    st.subheader("☁️ Palabras Frecuentes en Abstracts")

    st.info("""
    La nube de palabras permite identificar los conceptos más recurrentes presentes en los resúmenes de los artículos.
    """)

    texto = " ".join(
        df["Abstract"]
        .fillna("")
        .astype(str)
    )

    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=stopwords
    ).generate(texto)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(wordcloud)

    ax.axis("off")

    st.pyplot(fig)

# ==================================================
# TAB 3 - DATASET
# ==================================================

with tab3:

    st.subheader("Dataset Filtrado")

    st.write(
        f"Cantidad de registros mostrados: {len(df_filtrado)}"
    )

    st.dataframe(df_filtrado)

# ==================================================
# PIE DE PÁGINA
# ==================================================

st.markdown("---")

st.caption(
    "Proyecto académico desarrollado con Streamlit utilizando publicaciones científicas extraídas desde Scopus."
)
