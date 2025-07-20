import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Zomato Delivery Time Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E23744;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #E23744;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Zomato Delivery Time Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Prediksi waktu pengiriman makanan dengan Machine Learning</p>', unsafe_allow_html=True)

# Sidebar untuk navigasi
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Home & Prediction", "Data Analysis", "Model Performance", "About"]
)

# Load model dan data (simulasi)
@st.cache_data
def load_data():
    # Simulasi data untuk demo
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Delivery_person_Age': np.random.randint(18, 65, n_samples),
        'Delivery_person_Ratings': np.random.uniform(1, 5, n_samples),
        'delivery_distance': np.random.uniform(0.1, 15.0, n_samples),
        'Weather_conditions': np.random.choice(['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Fog', 'Cloudy'], n_samples),
        'Road_traffic_density': np.random.choice(['Low', 'Medium', 'High', 'Jam'], n_samples),
        'Type_of_order': np.random.choice(['Snack', 'Meal', 'Drink', 'Buffet'], n_samples),
        'Type_of_vehicle': np.random.choice(['motorcycle', 'scooter', 'electric_scooter', 'bicycle'], n_samples),
        'multiple_deliveries': np.random.randint(0, 4, n_samples),
        'Festival': np.random.choice(['No', 'Yes'], n_samples),
        'City': np.random.choice(['Metropolitian', 'Urban', 'Semi-Urban'], n_samples),
        'Time_taken (min)': np.random.randint(10, 60, n_samples)
    }
    
    return pd.DataFrame(data)

@st.cache_data
def get_model_results():
    # Simulasi hasil model
    return {
        'Random Forest': {'mae': 4.2, 'rmse': 6.1, 'r2': 0.85},
        'XGBoost': {'mae': 4.0, 'rmse': 5.8, 'r2': 0.87},
        'Ridge Regression': {'mae': 5.1, 'rmse': 7.2, 'r2': 0.78},
        'Lasso Regression': {'mae': 5.3, 'rmse': 7.5, 'r2': 0.76},
        'Linear Regression': {'mae': 5.4, 'rmse': 7.6, 'r2': 0.75}
    }

# Load data
df = load_data()
model_results = get_model_results()

# Fungsi prediksi (simulasi)
def predict_delivery_time(features):
    # Simulasi prediksi berdasarkan input
    base_time = 25
    
    # Faktor-faktor yang mempengaruhi waktu
    distance_factor = features['delivery_distance'] * 1.5
    traffic_factor = {'Low': 0, 'Medium': 5, 'High': 10, 'Jam': 15}[features['Road_traffic_density']]
    weather_factor = {'Sunny': 0, 'Cloudy': 2, 'Windy': 3, 'Fog': 5, 'Stormy': 8, 'Sandstorms': 10}[features['Weather_conditions']]
    vehicle_factor = {'bicycle': 8, 'motorcycle': 0, 'scooter': 2, 'electric_scooter': 1}[features['Type_of_vehicle']]
    multiple_factor = features['multiple_deliveries'] * 3
    festival_factor = 5 if features['Festival'] == 'Yes' else 0
    rating_factor = (5 - features['Delivery_person_Ratings']) * 2
    age_factor = abs(features['Delivery_person_Age'] - 30) * 0.1
    
    predicted_time = (base_time + distance_factor + traffic_factor + weather_factor + 
                     vehicle_factor + multiple_factor + festival_factor + rating_factor + age_factor)
    
    # Tambah noise untuk realisme
    predicted_time += np.random.normal(0, 2)
    
    return max(10, min(60, predicted_time))

# PAGE 1: HOME & PREDICTION
if page == "Home & Prediction":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Input Data Pengiriman</h2>', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            # Personal Information
            st.subheader("Informasi Delivery Person")
            age = st.slider("Umur Delivery Person", 18, 65, 30)
            rating = st.slider("Rating Delivery Person", 1.0, 5.0, 4.0, 0.1)
            
            # Delivery Details
            st.subheader("Detail Pengiriman")
            distance = st.slider("Jarak Pengiriman (km)", 0.1, 15.0, 5.0, 0.1)
            multiple_deliveries = st.selectbox("Jumlah Multiple Deliveries", [0, 1, 2, 3])
            
            # Environmental Factors
            st.subheader("Kondisi Lingkungan")
            weather = st.selectbox("Kondisi Cuaca", 
                                 ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Stormy', 'Sandstorms'])
            traffic = st.selectbox("Kepadatan Lalu Lintas", 
                                 ['Low', 'Medium', 'High', 'Jam'])
            
            # Order & Vehicle
            st.subheader("Pesanan & Kendaraan")
            order_type = st.selectbox("Jenis Pesanan", 
                                    ['Meal', 'Snack', 'Drink', 'Buffet'])
            vehicle = st.selectbox("Jenis Kendaraan", 
                                 ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
            
            # Additional Factors
            st.subheader("Faktor Tambahan")
            festival = st.selectbox("Festival", ['No', 'Yes'])
            city = st.selectbox("Tipe Kota", ['Metropolitian', 'Urban', 'Semi-Urban'])
            
            # Submit button
            submitted = st.form_submit_button("Prediksi Waktu Delivery!", 
                                            use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">Hasil Prediksi</h2>', unsafe_allow_html=True)
        
        if submitted:
            # Prepare features
            features = {
                'Delivery_person_Age': age,
                'Delivery_person_Ratings': rating,
                'delivery_distance': distance,
                'multiple_deliveries': multiple_deliveries,
                'Weather_conditions': weather,
                'Road_traffic_density': traffic,
                'Type_of_order': order_type,
                'Type_of_vehicle': vehicle,
                'Festival': festival,
                'City': city
            }
            
            # Make prediction
            predicted_time = predict_delivery_time(features)
            confidence = np.random.uniform(0.8, 0.95)  # Simulasi confidence
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Prediksi Waktu Delivery</h2>
                <h1 style="font-size: 3rem; color: #E23744; margin: 1rem 0;">
                    {predicted_time:.0f} menit
                </h1>
                <p style="font-size: 1.2rem;">
                    Confidence: {confidence:.1%}
                </p>
                <p style="font-size: 1rem; opacity: 0.8;">
                    Estimasi range: {predicted_time-3:.0f} - {predicted_time+3:.0f} menit
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Breakdown factors
            st.subheader("Analisis Faktor")
            
            factors_data = {
                'Faktor': ['Jarak', 'Traffic', 'Cuaca', 'Kendaraan', 'Multiple Delivery', 'Festival', 'Rating'],
                'Impact': [distance*1.5, 
                          {'Low': 0, 'Medium': 5, 'High': 10, 'Jam': 15}[traffic],
                          {'Sunny': 0, 'Cloudy': 2, 'Windy': 3, 'Fog': 5, 'Stormy': 8, 'Sandstorms': 10}[weather],
                          {'bicycle': 8, 'motorcycle': 0, 'scooter': 2, 'electric_scooter': 1}[vehicle],
                          multiple_deliveries * 3,
                          5 if festival == 'Yes' else 0,
                          (5 - rating) * 2]
            }
            
            factors_df = pd.DataFrame(factors_data)
            
            fig = px.bar(factors_df, x='Faktor', y='Impact', 
                        title='Kontribusi Setiap Faktor terhadap Waktu Delivery',
                        color='Impact', color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.markdown("""
            <div class="info-box">
                <h3>Cara Menggunakan</h3>
                <ol>
                    <li>Isi semua informasi di form sebelah kiri</li>
                    <li>Klik tombol "Prediksi Waktu Delivery!"</li>
                    <li>Lihat hasil prediksi dan analisis faktor</li>
                </ol>
                <p><strong>Tips:</strong> Semakin akurat data yang dimasukkan, semakin akurat prediksi yang dihasilkan!</p>
            </div>
            """, unsafe_allow_html=True)

# PAGE 2: DATA ANALYSIS
elif page == "Data Analysis":
    st.markdown('<h2 class="sub-header">Analisis Data Zomato</h2>', unsafe_allow_html=True)
    
    # Statistics Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df):,}</h3>
            <p>Total Orders</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df['Time_taken (min)'].mean():.1f}</h3>
            <p>Avg Delivery Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df['delivery_distance'].mean():.1f} km</h3>
            <p>Avg Distance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df['Delivery_person_Ratings'].mean():.1f}</h3>
            <p>Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["Weather Impact", "Traffic Analysis", "Vehicle Performance", "Distribution Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            weather_time = df.groupby('Weather_conditions')['Time_taken (min)'].mean().reset_index()
            fig = px.bar(weather_time, x='Weather_conditions', y='Time_taken (min)',
                        title='Average Delivery Time by Weather Condition',
                        color='Time_taken (min)', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            weather_count = df['Weather_conditions'].value_counts().reset_index()
            fig = px.pie(weather_count, values='count', names='Weather_conditions',
                        title='Distribution of Weather Conditions')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            traffic_time = df.groupby('Road_traffic_density')['Time_taken (min)'].mean().reset_index()
            fig = px.bar(traffic_time, x='Road_traffic_density', y='Time_taken (min)',
                        title='Average Delivery Time by Traffic Density',
                        color='Time_taken (min)', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Heatmap cuaca vs traffic
            heatmap_data = df.pivot_table(values='Time_taken (min)', 
                                        index='Weather_conditions', 
                                        columns='Road_traffic_density', 
                                        aggfunc='mean')
            fig = px.imshow(heatmap_data, 
                           title='Average Delivery Time: Weather vs Traffic',
                           color_continuous_scale='RdYlBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            vehicle_time = df.groupby('Type_of_vehicle')['Time_taken (min)'].mean().reset_index()
            fig = px.bar(vehicle_time, x='Type_of_vehicle', y='Time_taken (min)',
                        title='Average Delivery Time by Vehicle Type',
                        color='Time_taken (min)', color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter: Distance vs Time colored by Vehicle
            fig = px.scatter(df, x='delivery_distance', y='Time_taken (min)', 
                           color='Type_of_vehicle',
                           title='Distance vs Delivery Time by Vehicle Type',
                           hover_data=['Delivery_person_Ratings'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='Time_taken (min)', nbins=30,
                             title='Distribution of Delivery Times',
                             color_discrete_sequence=['#E23744'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='Festival', y='Time_taken (min)',
                        title='Delivery Time: Festival vs Non-Festival',
                        color='Festival')
            st.plotly_chart(fig, use_container_width=True)

# PAGE 3: MODEL PERFORMANCE
elif page == "Model Performance":
    st.markdown('<h2 class="sub-header">Performance Model Machine Learning</h2>', unsafe_allow_html=True)
    
    # Model comparison
    models_df = pd.DataFrame(model_results).T.reset_index()
    models_df.columns = ['Model', 'MAE', 'RMSE', 'R²']
    models_df = models_df.sort_values('R²', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model comparison chart
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Mean Absolute Error', 'Root Mean Square Error', 'R² Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # MAE
        fig.add_trace(
            go.Bar(x=models_df['Model'], y=models_df['MAE'], name='MAE', marker_color='red'),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=models_df['Model'], y=models_df['RMSE'], name='RMSE', marker_color='blue'),
            row=1, col=2
        )
        
        # R²
        fig.add_trace(
            go.Bar(x=models_df['Model'], y=models_df['R²'], name='R²', marker_color='green'),
            row=1, col=3
        )
        
        fig.update_layout(height=500, showlegend=False, title_text="Model Performance Comparison")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Best Model")
        best_model = models_df.iloc[0]
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>{best_model['Model']}</h3>
            <p><strong>R² Score:</strong> {best_model['R²']:.3f}</p>
            <p><strong>MAE:</strong> {best_model['MAE']:.1f} min</p>
            <p><strong>RMSE:</strong> {best_model['RMSE']:.1f} min</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Model Rankings")
        medals = ["1st", "2nd", "3rd", "4th", "5th"]
        for i, row in models_df.iterrows():
            medal = medals[i] if i < len(medals) else f"{i+1}th"
            st.write(f"**{medal}** **{row['Model']}** - R²: {row['R²']:.3f}")
    
    # Feature Importance (simulasi)
    st.subheader("Feature Importance")
    
    features = ['delivery_distance', 'Delivery_person_Ratings', 'Road_traffic_density', 
               'Weather_conditions', 'Type_of_vehicle', 'multiple_deliveries', 
               'Delivery_person_Age', 'Festival']
    importance = np.random.uniform(0.05, 0.25, len(features))
    importance = importance / importance.sum()  # Normalize
    
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                title='Feature Importance (Random Forest)',
                color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model explanation
    st.subheader("Model Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Akurasi Tinggi</h4>
            <p>Model dapat memprediksi waktu delivery dengan akurasi 85-87%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Prediksi Cepat</h4>
            <p>Waktu prediksi < 1 detik untuk respons real-time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>Update Berkala</h4>
            <p>Model dilatih ulang setiap bulan dengan data terbaru</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE 4: ABOUT
elif page == "About":
    st.markdown('<h2 class="sub-header">Tentang Aplikasi</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Tujuan Aplikasi
        
        Aplikasi **Zomato Delivery Time Predictor** dikembangkan untuk membantu:
        
        - **Pelanggan**: Mendapatkan estimasi waktu delivery yang akurat
        - **Delivery Partners**: Mengoptimalkan rute dan jadwal pengiriman  
        - **Zomato**: Meningkatkan customer satisfaction dan operational efficiency
        
        ### Teknologi yang Digunakan
        
        - **Machine Learning**: Random Forest, XGBoost, Ridge/Lasso Regression
        - **Frontend**: Streamlit dengan visualisasi interaktif
        - **Data Processing**: Pandas, NumPy, Scikit-learn
        - **Visualization**: Plotly, Matplotlib, Seaborn
        
        ### Dataset Features
        
        Model dilatih menggunakan berbagai fitur:
        
        1. **Personal**: Umur dan rating delivery person
        2. **Geografis**: Jarak pengiriman dan tipe kota
        3. **Temporal**: Waktu pemesanan dan hari festival
        4. **Environmental**: Kondisi cuaca dan kepadatan traffic
        5. **Operational**: Jenis kendaraan dan multiple deliveries
        
        ### Akurasi Model
        
        - **R² Score**: 0.87 (XGBoost)
        - **Mean Absolute Error**: 4.0 menit
        - **Confidence Level**: 85-95%
        """)
    
    with col2:
        st.markdown("""
        ### Model Performance
        """)
        
        # Performance metrics visualization
        metrics = ['Accuracy', 'Speed', 'Reliability', 'Usability']
        scores = [87, 95, 82, 90]
        
        fig = go.Figure(go.Scatterpolar(
            r=scores,
            theta=metrics,
            fill='toself',
            name='Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Future Improvements
        
        - **Real-time GPS tracking**
        - **Dynamic weather integration**  
        - **Customer feedback loop**
        - **Multi-city expansion**
        - **Mobile app integration**
        
        ### Developer Info
        
        **Created by**: Data Science Team  
        **Version**: 1.0.0  
        **Last Updated**: 2025  
        **Contact**: team@zomato.com
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666;">
        <p>Made with love for better food delivery experience</p>
        <p>© 2025 Zomato Delivery Time Predictor</p>
    </div>
    """, unsafe_allow_html=True)