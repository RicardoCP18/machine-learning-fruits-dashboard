import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# ---------------------------------------------------

# CONFIGURACIÓN DE LA PÁGINA

# ---------------------------------------------------

st.set_page_config(
page_title="Dashboard Bibliométrico",
page_icon="🍎",
layout="wide"
)

# ---------------------------------------------------

# CARGA DE DATOS

# ---------------------------------------------------

@st.cache_data
def cargar_datos():

```
url = "https://raw.githubusercontent.com/RicardoCP18/machine-learning-fruits-dashboard/refs/heads/main/data/scopus_PA3.csv"

return pd.read_csv(url)
```

df = cargar_datos()

# ---------------------------------------------------

# ENCABEZADO

# ---------------------------------------------------

st.title("🍎 Dashboard Bibliométrico sobre Machine Learning y Fruits")

st.markdown("""
Bienvenido al dashboard de análisis bibliométrico.

Esta aplicación explora publicaciones científicas obtenidas desde Scopus relacionadas con Machine Learning, clasificación de frutas y nutrición.

Los gráficos permiten identificar tendencias de investigación, autores destacados y conceptos frecuentes dentro de la literatura científica.
""")

# ---------------------------------------------------

# PREGUNTA DE INVESTIGACIÓN

# ---------------------------------------------------

st.subheader("Pregunta de Investigación")

st.info("""
¿Cómo se aplica el Machine Learning en la clasificación de frutas utilizando información nutricional para apoyar la toma de decisiones alimentarias saludables?
""")

# ---------------------------------------------------

# MÉTRICAS

# ---------------------------------------------------

st.subheader("Resumen General")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
"Artículos",
len(df)
)

col2.metric(
"Autores únicos",
len(set(";".join(df["Authors"].fillna("")).split(";")))
)

col3.metric(
"Años analizados",
df["Year"].nunique()
)

col4.metric(
"Total citas",
int(df["Cited by"].fillna(0).sum())
)

# ---------------------------------------------------

# DATASET

# ---------------------------------------------------

st.subheader("Vista Previa del Dataset")

st.dataframe(df)

# ---------------------------------------------------

# PUBLICACIONES POR AÑO

# ---------------------------------------------------

st.subheader("📈 Publicaciones por Año")

st.write("""
Este gráfico muestra la evolución de las publicaciones científicas relacionadas con la temática investigada.
""")

publicaciones = df["Year"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(
publicaciones.index,
publicaciones.values
)

ax.set_xlabel("Año")
ax.set_ylabel("Cantidad de publicaciones")

st.pyplot(fig)

# ---------------------------------------------------

# TOP 10 ARTÍCULOS MÁS CITADOS

# ---------------------------------------------------

st.subheader("🏆 Top 10 Artículos Más Citados")

st.write("""
Permite identificar los trabajos con mayor impacto académico.
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

fig, ax = plt.subplots(figsize=(12,6))

ax.barh(
top_citados["Titulo Corto"],
top_citados["Cited by"]
)

ax.set_xlabel("Número de citas")

plt.subplots_adjust(left=0.35)

st.pyplot(fig)

# ---------------------------------------------------

# AUTORES MÁS PRODUCTIVOS

# ---------------------------------------------------

st.subheader("👨‍🔬 Autores con Más Publicaciones")

st.write("""
Muestra los investigadores con mayor participación en el conjunto de artículos analizados.
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
columns=["Autor","Publicaciones"]
)

fig, ax = plt.subplots(figsize=(10,5))

ax.bar(
autores_df["Autor"],
autores_df["Publicaciones"]
)

plt.xticks(rotation=45)

st.pyplot(fig)

# ---------------------------------------------------

# REVISTAS CON MÁS PUBLICACIONES

# ---------------------------------------------------

st.subheader("📚 Revistas con Más Publicaciones")

st.write("""
Muestra las fuentes científicas con mayor cantidad de publicaciones sobre el tema.
""")

top_revistas = (
df["Source title"]
.value_counts()
.head(10)
)

fig, ax = plt.subplots(figsize=(10,5))

top_revistas.plot(
kind="bar",
ax=ax
)

plt.xticks(rotation=45)

st.pyplot(fig)

# ---------------------------------------------------

# TIPOS DE DOCUMENTOS

# ---------------------------------------------------

st.subheader("📄 Tipos de Documentos")

tipos = df["Document Type"].value_counts()

fig, ax = plt.subplots(figsize=(6,6))

ax.pie(
tipos,
labels=tipos.index,
autopct="%1.1f%%"
)

st.pyplot(fig)

# ---------------------------------------------------

# NUBE DE PALABRAS

# ---------------------------------------------------

st.subheader("☁️ Palabras Frecuentes en Abstracts")

st.write("""
La nube de palabras permite identificar los conceptos más recurrentes presentes en los resúmenes de los artículos.
""")

texto = " ".join(
df["Abstract"]
.fillna("")
.astype(str)
)

wordcloud = WordCloud(
width=1200,
height=600,
background_color="white"
).generate(texto)

fig, ax = plt.subplots(figsize=(12,6))

ax.imshow(wordcloud)

ax.axis("off")

st.pyplot(fig)

# ---------------------------------------------------

# CONCLUSIONES

# ---------------------------------------------------

st.subheader("Conclusiones")

st.success("""
La literatura científica relacionada con Machine Learning, clasificación de frutas y nutrición presenta una tendencia creciente.

Los artículos más citados evidencian el interés académico en técnicas de clasificación y análisis de datos aplicadas a la alimentación.

Los conceptos más frecuentes identificados en los abstracts muestran una fuerte relación entre aprendizaje automático, clasificación y nutrición.
""")

