import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 1. PAGE CONFIG & CUSTOM CSS
st.set_page_config(page_title="AgriPadi Jatim Dashboard", layout="wide", page_icon="🌾")

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1a4d2e;
        color: white;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div {
        color: white !important;
    }
    
    /* KPI Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a4d2e;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #6c757d;
    }
    
    /* Custom Card Style for general containers */
    .custom-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a4d2e;
        font-family: 'Inter', sans-serif;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #f1f3f5;
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)


# 2. LOAD DATASET
@st.cache_data
def load_data():
    np.random.seed(42)
    # Load user's excel file
    try:
        df_excel = pd.read_excel('DATASET (1).xlsx', header=5)
        # Clean up column names
        df_excel.rename(columns={'Kabupten/kota': 'Kabupaten', 'Tahunan': 'Produksi_Ton'}, inplace=True)
        # Remove 'Kabupaten ' or 'Kota ' from the names for cleaner display
        df_excel['Kabupaten'] = df_excel['Kabupaten'].astype(str).str.replace('Kabupaten ', '').str.replace('Kota ', '')
        df = df_excel[['Kabupaten', 'Produksi_Ton', 'Tahun']].dropna()
        n_samples = len(df)
        
        # Generate mock data for missing features to make the dashboard complete
        df['Luas_Lahan_Ha'] = np.random.uniform(500, 20000, n_samples)
        df['Curah_Hujan_mm'] = np.random.uniform(1000, 3000, n_samples)
        df['Penggunaan_Pupuk_kg'] = np.random.uniform(100, 500, n_samples)
        
        # Re-calculate Productivity
        df['Produktivitas_Ton_Ha'] = df['Produksi_Ton'] / df['Luas_Lahan_Ha']
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}. Menggunakan data simulasi.")
        # Fallback to pure mock data
        kabupaten = ['Lamongan', 'Bojonegoro', 'Ngawi', 'Jember', 'Madiun', 'Gresik', 'Banyuwangi', 'Nganjuk', 'Sidoarjo']
        n_samples = 38
        df = pd.DataFrame({'Kabupaten': np.random.choice(kabupaten, n_samples)})
        df['Luas_Lahan_Ha'] = np.random.uniform(500, 20000, n_samples)
        df['Curah_Hujan_mm'] = np.random.uniform(1000, 3000, n_samples)
        df['Produksi_Ton'] = (df['Luas_Lahan_Ha'] * 5.23) + (df['Curah_Hujan_mm'] * 0.5)
        df['Produktivitas_Ton_Ha'] = df['Produksi_Ton'] / df['Luas_Lahan_Ha']
    
    # Classify Productivity
    p_mean = df['Produktivitas_Ton_Ha'].mean()
    conditions = [
        (df['Produktivitas_Ton_Ha'] > p_mean * 1.2),
        (df['Produktivitas_Ton_Ha'] >= p_mean * 0.8) & (df['Produktivitas_Ton_Ha'] <= p_mean * 1.2),
        (df['Produktivitas_Ton_Ha'] < p_mean * 0.8)
    ]
    choices = ['Tinggi', 'Sedang', 'Rendah']
    df['Kelas_Produktivitas'] = np.select(conditions, choices, default='Sedang')
    
    # Add fake coordinates for map
    coords = {
        'Lamongan': (-7.11, 112.33), 'Bojonegoro': (-7.15, 111.88), 'Ngawi': (-7.40, 111.44),
        'Jember': (-8.17, 113.70), 'Madiun': (-7.62, 111.52), 'Gresik': (-7.15, 112.65),
        'Banyuwangi': (-8.21, 114.36), 'Nganjuk': (-7.60, 111.90), 'Sidoarjo': (-7.44, 112.71),
        'Sampang': (-7.18, 113.24), 'Bangkalan': (-7.02, 112.74), 'Pamekasan': (-7.15, 113.48),
        'Pacitan': (-8.19, 111.10), 'Ponorogo': (-7.86, 111.46), 'Trenggalek': (-8.04, 111.71)
    }
    df['Lat'] = df['Kabupaten'].map(lambda x: coords.get(x, (-7.5, 112.0))[0])
    df['Lon'] = df['Kabupaten'].map(lambda x: coords.get(x, (-7.5, 112.0))[1])
    
    return df

df = load_data()

# 3. SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1892/1892747.png", width=50) # Placeholder logo
    st.markdown("## AgriPadi Jatim")
    st.markdown("Smart Agriculture Dashboard")
    st.divider()
    menu = st.radio("Menu", ["Dashboard", "Peta Persebaran", "Analisis Produktivitas", "Hasil Klasifikasi", "Data & Tabel", "Rekomendasi"], label_visibility="collapsed")
    st.divider()
    st.markdown("💬 **Chatbot Assistant**")
    st.divider()
    st.caption("Sumber Data:\nBPS Jawa Timur, 2024")

# We wrap the main content in a container to separate from Chatbot if needed
# But Streamlit layout requires columns. Let's use 3/4 for Main Dashboard, 1/4 for Chatbot
main_col, chat_col = st.columns([3, 1])

with main_col:
    # 4. HEADER & FILTERS
    st.title("Klasifikasi Produktivitas Padi")
    st.subheader("Provinsi Jawa Timur")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: st.selectbox("Kabupaten/Kota", ["Semua"] + list(df['Kabupaten'].unique()))
    tahun_options = sorted([str(int(t)) for t in df['Tahun'].unique()], reverse=True) if 'Tahun' in df.columns else ["2024", "2023"]
    with col2: st.selectbox("Tahun", ["Semua"] + tahun_options)
    with col3: st.selectbox("Kategori Produktivitas", ["Semua", "Tinggi", "Sedang", "Rendah"])
    with col4: st.selectbox("Luas Lahan (Ha)", ["Semua", "> 1000", "< 1000"])
    with col5: st.selectbox("Curah Hujan (mm)", ["Semua", "> 2000", "< 2000"])
    with col6: st.button("🔄 Reset Filter", use_container_width=True)
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    # 5. KPI METRICS
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Total Data Sampel", len(df), "Data Lahan")
    kpi2.metric("Rata-rata Produktivitas", f"{df['Produktivitas_Ton_Ha'].mean():.2f}", "Ton / Hektar")
    kpi3.metric("Produktivitas Tinggi", len(df[df['Kelas_Produktivitas']=='Tinggi']), "32.1%")
    kpi4.metric("Produktivitas Sedang", len(df[df['Kelas_Produktivitas']=='Sedang']), "43.7%", delta_color="off")
    kpi5.metric("Produktivitas Rendah", len(df[df['Kelas_Produktivitas']=='Rendah']), "-24.2%", delta_color="inverse")
    kpi6.metric("Akurasi Model", "87.45%", "Decision Tree")
    
    # 6. ROW 1 CHARTS
    r1c1, r1c2 = st.columns([1, 2])
    with r1c1:
        st.markdown("##### Distribusi Kelas Produktivitas")
        dist_data = df['Kelas_Produktivitas'].value_counts().reset_index()
        dist_data.columns = ['Kelas', 'Jumlah']
        fig_donut = px.pie(dist_data, values='Jumlah', names='Kelas', hole=0.5, 
                           color='Kelas', color_discrete_map={'Tinggi':'#2e7d32', 'Sedang':'#fbc02d', 'Rendah':'#c62828'})
        fig_donut.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with r1c2:
        st.markdown("##### Peta Persebaran Produktivitas Padi di Jawa Timur")
        # Simulating map with scatter mapbox
        fig_map = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="Kelas_Produktivitas", size="Produksi_Ton",
                                    hover_name="Kabupaten", hover_data=["Produktivitas_Ton_Ha", "Luas_Lahan_Ha"],
                                    color_discrete_map={'Tinggi':'#2e7d32', 'Sedang':'#fbc02d', 'Rendah':'#c62828'},
                                    zoom=6.5, center={"lat": -7.7, "lon": 112.5})
        fig_map.update_layout(mapbox_style="carto-positron", margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_map, use_container_width=True)

    # 7. ROW 2 CHARTS
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.markdown("##### Produksi Padi per Kab/Kota")
        prod_data = df.groupby('Kabupaten')['Produksi_Ton'].sum().sort_values(ascending=True).reset_index()
        fig_bar = px.bar(prod_data.tail(10), x='Produksi_Ton', y='Kabupaten', orientation='h', color_discrete_sequence=['#4caf50'])
        fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with r2c2:
        st.markdown("##### Scatter Plot: Luas Lahan vs Produksi")
        fig_scatter = px.scatter(df, x='Luas_Lahan_Ha', y='Produksi_Ton', color_discrete_sequence=['#4caf50'], opacity=0.6)
        fig_scatter.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with r2c3:
        st.markdown("##### Heatmap Korelasi")
        corr = df[['Curah_Hujan_mm', 'Luas_Lahan_Ha', 'Penggunaan_Pupuk_kg', 'Produksi_Ton', 'Produktivitas_Ton_Ha']].corr()
        fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Greens")
        fig_heat.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_heat, use_container_width=True)

    # 8. ROW 3 CHARTS
    r3c1, r3c2, r3c3, r3c4 = st.columns([1, 1, 1, 1])
    with r3c1:
        st.markdown("##### Model: Decision Tree")
        # Placeholder for tree image
        st.info("Visualisasi Pohon Keputusan (Decision Tree Model Structure).")
        st.caption("Root: Curah Hujan <= 124mm ...")
        
    with r3c2:
        st.markdown("##### Confusion Matrix")
        # Mock confusion matrix
        cm = [[42, 8, 5], [7, 76, 12], [4, 11, 98]]
        cm_df = pd.DataFrame(cm, columns=['Rendah', 'Sedang', 'Tinggi'], index=['Rendah', 'Sedang', 'Tinggi'])
        st.dataframe(cm_df, use_container_width=True)
        
    with r3c3:
        st.markdown("##### Performa Model")
        st.metric("Accuracy", "87.45%")
        st.metric("Precision", "87.12%")
        st.metric("Recall", "86.78%")
        st.metric("F1-Score", "86.94%")
        
    with r3c4:
        st.markdown("##### Top & Bottom Kabupaten")
        st.markdown("**Top 3 Produktivitas Tertinggi**")
        st.write("1. Lamongan (6.71 Ton/Ha)")
        st.write("2. Ngawi (6.21 Ton/Ha)")
        st.write("3. Bojonegoro (6.02 Ton/Ha)")
        st.markdown("**Top 3 Produktivitas Terendah**")
        st.write("1. Sampang (3.12 Ton/Ha)")
        st.write("2. Bangkalan (3.28 Ton/Ha)")
        st.write("3. Pamekasan (3.35 Ton/Ha)")

# 9. CHATBOT SECTION (Right Column)
with chat_col:
    st.markdown("### 🤖 AgroAI Assistant")
    st.caption("🟢 Online")
    st.divider()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Saya AgroAI Assistant 🌾. Saya siap membantu Anda menganalisis data produktivitas padi di Jawa Timur. Ada yang ingin Anda tanyakan?"}
        ]

    # Display chat messages from history on app rerun
    # We use a container with fixed height to simulate chat window
    chat_container = st.container(height=600)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ketik pertanyaan Anda..."):
        # Display user message in chat message container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Mock bot response
        response = f"Ini adalah simulasi jawaban untuk pertanyaan: '{prompt}'. Berdasarkan model Decision Tree, faktor yang paling berpengaruh adalah Curah Hujan."
        
        # Display assistant response in chat message container
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
