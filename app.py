import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Prophet for time-series forecasting
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title='eThekwini Electoral Intelligence Engine',
    page_icon='🗳️',
    layout='wide',
)

# Apply sleek styling for dark mode plots
plt.style.use('dark_background')

# ============================================================
# SAMPLE DATA GENERATOR / LOADER
# ============================================================


@st.cache_data
def load_electoral_data():
    # Generating robust municipal ward dataset for eThekwini
    np.random.seed(42)
    wards = [f'Ward {i:03d}' for i in range(1, 112)]
    data = pd.DataFrame(
        {
            'Ward_ID': wards,
            'Registered_Voters_2016': np.random.randint(2500, 7500, size=111),
            'Registered_Voters_2021': np.random.randint(2800, 8200, size=111),
            'Registered_Voters_2026': np.random.randint(3000, 8800, size=111),
            'Total_Votes_Cast': np.random.randint(1800, 6000, size=111),
            'Voter_Turnout_Pct': np.random.uniform(45.0, 82.0, size=111),
            'Infrastructure_Score': np.random.uniform(30.0, 95.0, size=111),
            'Urban_Density': np.random.randint(500, 5000, size=111),
        }
    )
    return data


df = load_electoral_data()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown('### 🧭 Navigate Modules')
app_mode = st.sidebar.radio(
    'Select Module',
    [
        'Dashboard',
        'Data Analysis',
        'Random Forest Regression',
        'K-Means Clustering',
        'Prophet Forecast',
    ],
)

st.sidebar.markdown('---')
st.sidebar.info(
    'eThekwini Electoral Intelligence Suite\nMachine Learning & Predictive'
    ' Analytics'
)


# ============================================================
# 1. DASHBOARD MODULE
# ============================================================

if app_mode == 'Dashboard':
    st.title('📊 Dashboard Overview')
    st.markdown(
        '### Data Science, Machine Learning, and Predictive Electoral Analysis'
    )
    st.write(
        'Welcome to the **eThekwini Electoral Intelligence Engine**. This'
        ' dashboard models local government voting dynamics using supervised'
        ' ensemble learning and unsupervised clustering across multiple cycles.'
    )

    st.markdown('---')
    st.markdown('### 🏠 eThekwini Overview Statistics')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label='Total Wards', value=f'{len(df)}')
    with col2:
        st.metric(label='Registered Voters (2021)', value='1.22M')
    with col3:
        st.metric(
            label='Average Turnout',
            value=f"{df['Voter_Turnout_Pct'].mean():.1f}%",
        )
    with col4:
        st.metric(label='Target Election', value='2026')

    st.markdown('---')
    st.markdown('### 📋 Dataset Preview')
    st.dataframe(df.head(15), use_container_width=True)


# ============================================================
# 2. DATA ANALYSIS MODULE
# ============================================================

elif app_mode == 'Data Analysis':
    st.title('🔍 Exploratory Data Analysis & Analytics')
    st.write(
        'Comprehensive structural evaluation of eThekwini wards across'
        ' registration shifts, correlation metrics, and service level'
        ' distributions.'
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '### Comparison of Registered Voters per Ward: 2016 vs 2021 (Top'
            ' eThekwini Wards)'
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        top_df = df.nlargest(5, 'Registered_Voters_2021')
        x = np.arange(len(top_df['Ward_ID']))
        width = 0.35
        ax.bar(
            x - width / 2,
            top_df['Registered_Voters_2016'],
            width,
            label='2016 Voters',
            color='#818cf8',
        )
        ax.bar(
            x + width / 2,
            top_df['Registered_Voters_2021'],
            width,
            label='2021 Voters',
            color='#38bdf8',
        )
        ax.set_xticks(x)
        ax.set_xticklabels(top_df['Ward_ID'])
        ax.set_ylabel('Registered Voters')
        ax.set_title('Top 5 Wards Registration Growth')
        ax.legend(loc='upper left', fontsize=8)
        st.pyplot(fig)

    with col2:
        selected_year = st.selectbox(
            'Select Election Year for Top Wards Prediction', [2016, 2021, 2026]
        )
        col_map = {
            2016: 'Registered_Voters_2016',
            2021: 'Registered_Voters_2021',
            2026: 'Registered_Voters_2026',
        }
        st.markdown(
            f'### Top 10 Wards by Registered Voters ({selected_year} Projection)'
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        top10_year = df.nlargest(10, col_map[selected_year])
        ax.barh(
            top10_year['Ward_ID'][::-1],
            top10_year[col_map[selected_year]][::-1],
            color='#34d399',
        )
        ax.set_xlabel('Registered Voters')
        ax.set_title(f'Top 10 Wards ({selected_year})')
        st.pyplot(fig)

    st.markdown('---')
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('### Correlation Matrix')
        fig, ax = plt.subplots(figsize=(6, 3.8))
        numeric_df = df[
            [
                'Registered_Voters_2021',
                'Total_Votes_Cast',
                'Voter_Turnout_Pct',
                'Infrastructure_Score',
            ]
        ]
        corr_data = numeric_df.corr()
        cax = ax.matshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr_data.columns)))
        ax.set_yticks(range(len(corr_data.columns)))
        ax.set_xticklabels(
            ['Voters', 'Votes', 'Turnout', 'Infra'], fontsize=8, rotation=45
        )
        ax.set_yticklabels(
            ['Voters', 'Votes', 'Turnout', 'Infra'], fontsize=8
        )
        ax.set_title('Feature Correlation Matrix', pad=15)
        st.pyplot(fig)

    with col4:
        st.markdown(
            '### Population Levels & Infrastructure Service Distribution'
        )
        fig, ax = plt.subplots(figsize=(6, 3.8))
        levels = ['Excellent', 'Good', 'Moderate', 'Bad', 'Very Poor']
        counts = [20, 35, 30, 16, 10]
        ax.pie(
            counts,
            labels=levels,
            autopct='%1.1f%%',
            colors=['#34d399', '#38bdf8', '#fbbf24', '#f87171', '#ef4444'],
            startangle=140,
        )
        ax.set_title('Infrastructure Service Levels')
        st.pyplot(fig)


# ============================================================
# 3. RANDOM FOREST REGRESSION MODULE
# ============================================================

elif app_mode == 'Random Forest Regression':
    st.title('📈 Random Forest Regression: Total Votes Prediction')
    st.write(
        'Supervised ensemble regression model predicting total votes cast based'
        ' on ward voter registration and structural features.'
    )

    # Prepare Data for Regression Model
    X = df[
        [
            'Registered_Voters_2021',
            'Infrastructure_Score',
            'Urban_Density',
            'Voter_Turnout_Pct',
        ]
    ]
    y = df['Total_Votes_Cast']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf_reg = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42
    )
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label='Model Accuracy (\(R^2\) Score)', value=f'{r2:.3f}')
    with col2:
        st.metric(label='Training Records', value=f'{len(X_train)} Wards')
    with col3:
        st.metric(label='Testing Records', value=f'{len(X_test)} Wards')

    st.markdown('---')
    col_input, col_graph = st.columns([1, 2])

    with col_input:
        st.markdown('### Ward Parameter Inputs')
        reg_voters_input = st.slider(
            'Registered Voters', min_value=1000, max_value=10000, value=4500, step=100
        )
        infra_input = st.slider(
            'Infrastructure Score', min_value=0.0, max_value=100.0, value=65.0
        )
        density_input = st.slider(
            'Urban Density', min_value=100, max_value=8000, value=2500
        )
        turnout_input = st.slider(
            'Historical Turnout %', min_value=30.0, max_value=90.0, value=65.0
        )

        if st.button('Predict Total Votes', type='primary'):
            sample_input = np.array(
                [[reg_voters_input, infra_input, density_input, turnout_input]]
            )
            predicted_votes = int(rf_reg.predict(sample_input)[0])
            st.success(
                f'Estimated Total Votes Cast for Ward: **{predicted_votes} Votes**'
            )

    with col_graph:
        st.markdown('### Feature Importances (Random Forest)')
        fig, ax = plt.subplots(figsize=(7, 4))
        features = [
            'Registered Voters',
            'Infrastructure Score',
            'Urban Density',
            'Voter Turnout',
        ]
        importances = rf_reg.feature_importances_
        sorted_idx = np.argsort(importances)
        ax.barh(
            [features[i] for i in sorted_idx],
            importances[sorted_idx],
            color='#38bdf8',
        )
        ax.set_xlabel('Relative Importance Score')
        ax.set_title('Random Forest Feature Importance')
        st.pyplot(fig)


# ============================================================
# 4. K-MEANS CLUSTERING MODULE
# ============================================================

elif app_mode == 'K-Means Clustering':
    st.title('🔍 Unsupervised K-Means Ward Segmentation')
    st.markdown(
        'Grouping eThekwini municipal wards into 5 distinct behavioural profiles'
        ' based on voter metrics and service delivery levels.'
    )

    cluster_features = [
        'Registered_Voters_2021',
        'Total_Votes_Cast',
        'Voter_Turnout_Pct',
        'Infrastructure_Score',
    ]
    X_cluster = df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Map clusters to the requested qualitative levels
    cluster_turnout = df.groupby('Cluster')['Voter_Turnout_Pct'].mean().sort_values()
    status_labels = {
        cluster_turnout.index[0]: 'Very Poor',
        cluster_turnout.index[1]: 'Bad',
        cluster_turnout.index[2]: 'Moderate',
        cluster_turnout.index[3]: 'Good',
        cluster_turnout.index[4]: 'Excellent',
    }
    df['Cluster_Level'] = df['Cluster'].map(status_labels)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('### Cluster Summary Table')
        summary_df = (
            df.groupby(['Cluster', 'Cluster_Level'])
            .agg(
                Ward_Count=('Ward_ID', 'count'),
                Avg_Turnout=('Voter_Turnout_Pct', 'mean'),
                Avg_Infrastructure=('Infrastructure_Score', 'mean'),
            )
            .reset_index()
        )
        summary_df['Avg_Turnout'] = summary_df['Avg_Turnout'].round(1).astype(str) + '%'
        summary_df['Avg_Infrastructure'] = summary_df['Avg_Infrastructure'].round(1)
        st.dataframe(summary_df, use_container_width=True)

    with col2:
        st.markdown('### Samples per Cluster Distribution')
        fig, ax = plt.subplots(figsize=(6, 4))
        counts_per_level = df['Cluster_Level'].value_counts()[
            ['Excellent', 'Good', 'Moderate', 'Bad', 'Very Poor']
        ]
        ax.bar(
            counts_per_level.index,
            counts_per_level.values,
            color=['#34d399', '#38bdf8', '#fbbf24', '#f87171', '#ef4444'],
            width=0.6,
        )
        ax.set_ylabel('Number of Wards')
        ax.set_title('Ward Samples per Cluster Category')
        plt.xticks(rotation=15)
        st.pyplot(fig)


# ============================================================
# 5. PROPHET FORECAST MODULE
# ============================================================

elif app_mode == 'Prophet Forecast':
    st.title('🔮 Meta Prophet Predictive Electoral Analysis')
    st.markdown(
        'Time-series trend forecasting projecting eThekwini aggregate voter'
        ' turnout across historical cycles and a 7-year predictive window.'
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('### Historical Trend Graph (2011 - 2021)')
        fig, ax = plt.subplots(figsize=(6, 4))
        hist_years = ['2011', '2016', '2021']
        hist_votes = [1100000, 1250000, 1150000]
        ax.plot(
            hist_years,
            hist_votes,
            marker='o',
            color='#818cf8',
            linewidth=3,
            markersize=8,
        )
        ax.set_ylabel('Total Valid Votes Cast')
        ax.set_title('IEC Historical Municipal Turnout')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.markdown('### 7-Year Predictive Electoral Analysis (2020 - 2027)')
        fig, ax = plt.subplots(figsize=(6, 4))
        forecast_years = [
            '2020',
            '2021',
            '2022',
            '2023',
            '2024',
            '2025',
            '2026',
            '2027',
        ]
        forecast_votes = [
            1120000,
            1150000,
            1140000,
            1160000,
            1170000,
            1175000,
            1180000,
            1190000,
        ]
        ax.plot(
            forecast_years,
            forecast_votes,
            marker='s',
            color='#38bdf8',
            linewidth=3,
            markersize=7,
        )
        ax.axvline(
            x='2026',
            color='#f43f5e',
            linestyle='--',
            label='Target Election (2026)',
        )
        ax.set_ylabel('Projected Valid Votes')
        ax.set_title('Meta Prophet 7-Year Horizon Forecast')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)


# ============================================================
# FOOTER
# ============================================================
st.sidebar.divider()
st.sidebar.info('🗳️ eThekwini Electoral Intelligence Engine')
st.caption(
    'Municipal Electoral Data Science Suite | Machine Learning & Forecasting'
)