
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="Earthquake Clustering Dashboard",
    page_icon="🌍",
    layout="wide"
)

sns.set_style("whitegrid")

# =====================================================
# MEMBACA DATASET
# =====================================================

DATA_PATH = Path("global_earthquakes_seismic.csv")

if not DATA_PATH.exists():
    st.error("Dataset tidak ditemukan. Periksa folder dataset!")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Mengambil fitur numerik
numeric_df = df.select_dtypes(include=np.number).copy()

# Mendeteksi dan menghapus kolom ID
id_columns = [
    col for col in numeric_df.columns
    if any(x in col.lower() for x in ["id", "index", "unnamed"])
]

numeric_df.drop(columns=id_columns, inplace=True)

# Mengubah nilai tak hingga menjadi NaN
numeric_df = numeric_df.replace(
    [np.inf, -np.inf], np.nan
)

# Menghapus kolom kosong seluruhnya
numeric_df.dropna(axis=1, how="all", inplace=True)

if numeric_df.shape[1] < 2:
    st.error("Dataset membutuhkan minimal dua fitur numerik.")
    st.stop()

# =====================================================
# PREPROCESSING
# =====================================================

# Menghapus data duplikat
numeric_df = numeric_df.drop_duplicates()

# Mengisi missing values
imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(numeric_df),
    columns=numeric_df.columns,
    index=numeric_df.index
)

# Normalisasi data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

# =====================================================
# HEADER DASHBOARD
# =====================================================

st.title("🌍 Earthquake Clustering Dashboard")

st.markdown("""
Dashboard ini menampilkan analisis dan visualisasi data gempa bumi
menggunakan algoritma **K-Means Clustering**.

Pengelompokan dilakukan berdasarkan karakteristik fitur numerik
pada dataset Global Earthquakes Seismic Telemetry.
""")

st.divider()

# =====================================================
# RINGKASAN DATASET
# =====================================================

st.header("1. Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Jumlah Data",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Jumlah Kolom",
        df.shape[1]
    )

with col3:
    st.metric(
        "Fitur Numerik",
        numeric_df.shape[1]
    )

with col4:
    st.metric(
        "Data Duplikat Dihapus",
        len(df) - len(numeric_df)
    )

st.subheader("Preview Dataset")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.subheader("Statistik Deskriptif")

st.dataframe(
    numeric_df.describe().T,
    use_container_width=True
)

# =====================================================
# INFORMASI FITUR
# =====================================================

st.subheader("Informasi Fitur yang Digunakan")

st.write(
    "Fitur numerik yang digunakan dalam proses clustering:"
)

st.write(", ".join(numeric_df.columns.tolist()))

st.divider()

# =====================================================
# EDA - DISTRIBUSI FITUR
# =====================================================

st.header("2. Exploratory Data Analysis (EDA)")

st.subheader("Distribusi Fitur Numerik")

selected_distribution = st.selectbox(
    "Pilih fitur untuk melihat distribusi:",
    numeric_df.columns.tolist()
)

fig, ax = plt.subplots(figsize=(9, 4))

sns.histplot(
    numeric_df[selected_distribution],
    bins=30,
    kde=True,
    ax=ax
)

ax.set_title(
    f"Distribusi {selected_distribution}"
)

ax.set_xlabel(selected_distribution)
ax.set_ylabel("Frekuensi")

st.pyplot(fig)
plt.close(fig)

st.info("""
**Analisis Distribusi:**
Histogram memperlihatkan persebaran nilai pada fitur yang dipilih.
Grafik ini membantu mengetahui apakah data terkonsentrasi pada
rentang tertentu, memiliki distribusi tidak merata, atau memiliki
nilai ekstrem.
""")

# =====================================================
# BOXPLOT
# =====================================================

st.subheader("Deteksi Outlier")

selected_boxplot = st.selectbox(
    "Pilih fitur untuk melihat boxplot:",
    numeric_df.columns.tolist(),
    key="boxplot"
)

fig, ax = plt.subplots(figsize=(9, 3))

sns.boxplot(
    x=numeric_df[selected_boxplot],
    ax=ax
)

ax.set_title(
    f"Boxplot {selected_boxplot}"
)

st.pyplot(fig)
plt.close(fig)

st.info("""
**Analisis Boxplot:**
Boxplot digunakan untuk melihat persebaran data dan mendeteksi
kemungkinan outlier. Nilai ekstrem perlu diperiksa terlebih dahulu
karena pada data gempa nilai tersebut dapat menggambarkan kejadian
gempa yang sebenarnya.
""")

# =====================================================
# CORRELATION HEATMAP
# =====================================================

st.subheader("Korelasi Antarfitur")

corr = numeric_df.corr()

fig, ax = plt.subplots(
    figsize=(11, 7)
)

sns.heatmap(
    corr,
    annot=len(corr.columns) <= 15,
    cmap="coolwarm",
    fmt=".2f",
    center=0,
    ax=ax
)

ax.set_title("Correlation Heatmap")

st.pyplot(fig)
plt.close(fig)

st.info("""
**Analisis Korelasi:**
Heatmap menunjukkan hubungan antarfitur numerik. Nilai korelasi
positif menunjukkan hubungan searah, sedangkan nilai negatif
menunjukkan hubungan berlawanan arah. Korelasi tidak menunjukkan
hubungan sebab-akibat.
""")

st.divider()

# =====================================================
# PENGATURAN CLUSTER
# =====================================================

st.header("3. Pengaturan K-Means Clustering")

k_value = st.slider(
    "Pilih Jumlah Cluster (K)",
    min_value=2,
    max_value=min(10, len(X_scaled) - 1),
    value=3
)

st.write(
    f"Jumlah cluster yang dipilih: **{k_value}**"
)

# =====================================================
# ELBOW METHOD
# =====================================================

st.subheader("Elbow Method")

K_range = range(
    2,
    min(11, len(X_scaled))
)

inertia = []

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    list(K_range),
    inertia,
    marker="o"
)

ax.set_title("Elbow Method")
ax.set_xlabel("Jumlah Cluster (K)")
ax.set_ylabel("Inertia")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.info("""
**Analisis Elbow Method:**
Grafik digunakan untuk mencari titik siku ketika penurunan inertia
mulai melambat. Titik tersebut menjadi kandidat jumlah cluster
yang sesuai untuk model K-Means.
""")

# =====================================================
# SILHOUETTE SCORE
# =====================================================

st.subheader("Silhouette Score")

silhouette_values = []

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_values.append(score)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    list(K_range),
    silhouette_values,
    marker="o"
)

ax.set_title("Silhouette Score")
ax.set_xlabel("Jumlah Cluster (K)")
ax.set_ylabel("Silhouette Score")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.info("""
**Analisis Silhouette Score:**
Silhouette Score digunakan untuk mengukur kualitas pengelompokan.
Nilai yang mendekati 1 menunjukkan cluster semakin kompak dan
terpisah dengan baik, sedangkan nilai negatif dapat menunjukkan
adanya data yang kurang sesuai dengan cluster-nya.
""")

st.divider()

# =====================================================
# MODEL K-MEANS
# =====================================================

st.header("4. Hasil Clustering K-Means")

kmeans = KMeans(
    n_clusters=k_value,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled)

result = X_imputed.copy()

result["cluster"] = labels

final_inertia = kmeans.inertia_

final_silhouette = silhouette_score(
    X_scaled,
    labels
)

# =====================================================
# METRIK EVALUASI
# =====================================================

st.subheader("Evaluasi Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Jumlah Cluster",
        k_value
    )

with col2:
    st.metric(
        "Inertia",
        f"{final_inertia:,.2f}"
    )

with col3:
    st.metric(
        "Silhouette Score",
        f"{final_silhouette:.4f}"
    )

evaluation_df = pd.DataFrame({
    "Metrik": [
        "Jumlah Cluster",
        "Inertia",
        "Silhouette Score"
    ],
    "Nilai": [
        k_value,
        round(final_inertia, 4),
        round(final_silhouette, 4)
    ]
})

st.dataframe(
    evaluation_df,
    use_container_width=True,
    hide_index=True
)

st.info("""
**Analisis Evaluasi:**
Inertia menunjukkan jumlah kuadrat jarak data terhadap pusat
cluster. Silhouette Score menunjukkan tingkat kekompakan dan
pemisahan antarcluster. Kedua metrik digunakan bersama untuk
menilai hasil pengelompokan.
""")

# =====================================================
# VISUALISASI SCATTER PLOT
# =====================================================

st.subheader("Visualisasi Hasil Clustering")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "Pilih fitur sumbu X:",
        numeric_df.columns.tolist(),
        index=0
    )

with col2:
    y_feature = st.selectbox(
        "Pilih fitur sumbu Y:",
        numeric_df.columns.tolist(),
        index=1
    )

fig = px.scatter(
    result,
    x=x_feature,
    y=y_feature,
    color="cluster",
    color_continuous_scale="Viridis",
    title="Visualisasi Hasil K-Means Clustering",
    hover_data=["cluster"]
)

fig.update_layout(
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info("""
**Analisis Scatter Plot:**
Setiap titik menggambarkan satu data gempa, sedangkan warna
menunjukkan kelompok hasil clustering. Pengguna dapat mengganti
fitur sumbu X dan Y untuk melihat pola pengelompokan dari sudut
pandang yang berbeda.
""")

# =====================================================
# JUMLAH ANGGOTA CLUSTER
# =====================================================

st.subheader("Distribusi Anggota Cluster")

cluster_count = result["cluster"].value_counts().sort_index()

fig = px.bar(
    x=cluster_count.index.astype(str),
    y=cluster_count.values,
    labels={
        "x": "Cluster",
        "y": "Jumlah Data"
    },
    title="Jumlah Data pada Setiap Cluster"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info("""
**Analisis Distribusi Cluster:**
Grafik batang memperlihatkan jumlah data pada setiap kelompok.
Perbedaan jumlah anggota menunjukkan bagaimana data tersebar
setelah proses clustering dilakukan.
""")

st.divider()

# =====================================================
# KARAKTERISTIK CLUSTER
# =====================================================

st.header("5. Karakteristik Setiap Cluster")

cluster_summary = result.groupby(
    "cluster"
)[numeric_df.columns].mean()

st.subheader("Tabel Rata-rata Karakteristik Cluster")

st.dataframe(
    cluster_summary.round(3),
    use_container_width=True
)

st.subheader("Visualisasi Karakteristik Cluster")

fig = px.bar(
    cluster_summary.reset_index().melt(
        id_vars="cluster",
        var_name="Fitur",
        value_name="Rata-rata"
    ),
    x="Fitur",
    y="Rata-rata",
    color="cluster",
    barmode="group",
    title="Perbandingan Rata-rata Fitur Setiap Cluster"
)

fig.update_layout(
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info("""
**Analisis Karakteristik Cluster:**
Tabel dan grafik menunjukkan nilai rata-rata fitur pada setiap
kelompok. Perbandingan tersebut dapat digunakan untuk mengetahui
perbedaan karakteristik masing-masing cluster, misalnya berdasarkan
magnitudo, kedalaman, atau koordinat gempa jika fitur tersebut
tersedia dalam dataset.
""")

# =====================================================
# DATA HASIL CLUSTERING
# =====================================================

st.subheader("Data Hasil Clustering")

st.dataframe(
    result,
    use_container_width=True
)

# =====================================================
# DOWNLOAD HASIL
# =====================================================

st.subheader("Download Hasil Analisis")

csv = result.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Hasil Clustering CSV",
    data=csv,
    file_name="hasil_clustering_gempa.csv",
    mime="text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center; color:gray;">

**Earthquake Clustering Dashboard**

Analisis dan Visualisasi Clustering Menggunakan Algoritma K-Means

Dataset: Global Earthquakes Seismic Telemetry

</div>
""", unsafe_allow_html=True)