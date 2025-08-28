"""
Advanced Customer Segmentation Dashboard
Interactive Streamlit Application for Customer Segmentation Analysis

Author: [Your Name]
Course: BMCS2003 Artificial Intelligence
Assignment: Machine Learning (Unsupervised)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the shopping trends data"""
    try:
        # Load data
        df = pd.read_csv('shopping_trends.csv')
        
        # Feature engineering
        frequency_mapping = {
            'Weekly': 52, 'Bi-Weekly': 26, 'Fortnightly': 26, 
            'Monthly': 12, 'Quarterly': 4, 'Annually': 1
        }
        
        # Create RFM features
        df['Recency_Score'] = df['Review Rating']
        df['Annual_Frequency'] = df['Frequency of Purchases'].map(frequency_mapping)
        df['Total_Purchase_Frequency'] = df['Previous Purchases'] * df['Annual_Frequency']
        df['Monetary_Score'] = df['Purchase Amount (USD)']
        df['CLV_Proxy'] = (df['Purchase Amount (USD)'] * df['Previous Purchases'] * df['Annual_Frequency']) / 100
        
        # Behavioral features
        df['Is_Subscribed'] = (df['Subscription Status'] == 'Yes').astype(int)
        df['Uses_Discounts'] = (df['Discount Applied'] == 'Yes').astype(int)
        df['Uses_Promos'] = (df['Promo Code Used'] == 'Yes').astype(int)
        df['Gender_Numeric'] = (df['Gender'] == 'Male').astype(int)
        
        # Select features for clustering
        clustering_features = [
            'Age', 'Recency_Score', 'Total_Purchase_Frequency', 'Monetary_Score',
            'CLV_Proxy', 'Is_Subscribed', 'Uses_Discounts', 'Uses_Promos',
            'Previous Purchases', 'Annual_Frequency', 'Gender_Numeric'
        ]
        
        X = df[clustering_features].fillna(df[clustering_features].median())
        
        return df, X, clustering_features
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_data
def perform_clustering(X, algorithm, n_clusters=5, **kwargs):
    """Perform clustering with specified algorithm"""
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algorithm == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == 'DBSCAN':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'Gaussian Mixture':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    labels = model.fit_predict(X_scaled)
    
    # Calculate evaluation metrics
    if len(set(labels)) > 1 and not (-1 in labels and len(set(labels)) == 2):
        try:
            if -1 in labels:  # DBSCAN with noise
                mask = labels != -1
                if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                    sil_score = silhouette_score(X_scaled[mask], labels[mask])
                    ch_score = calinski_harabasz_score(X_scaled[mask], labels[mask])
                    db_score = davies_bouldin_score(X_scaled[mask], labels[mask])
                else:
                    sil_score = ch_score = db_score = 0
            else:
                sil_score = silhouette_score(X_scaled, labels)
                ch_score = calinski_harabasz_score(X_scaled, labels)
                db_score = davies_bouldin_score(X_scaled, labels)
        except:
            sil_score = ch_score = db_score = 0
    else:
        sil_score = ch_score = db_score = 0
    
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    return labels, X_scaled, X_pca, sil_score, ch_score, db_score, pca.explained_variance_ratio_

def create_cluster_visualization(X_pca, labels, algorithm_name):
    """Create cluster visualization using PCA"""
    
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1], 
        color=labels.astype(str),
        title=f'{algorithm_name} Clustering Results (PCA Visualization)',
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        width=800, height=600,
        showlegend=True,
        legend_title="Cluster"
    )
    
    return fig

def create_cluster_characteristics(df, labels):
    """Create cluster characteristics analysis"""
    
    df_analysis = df.copy()
    df_analysis['Cluster'] = labels
    
    # Remove noise points
    if -1 in labels:
        df_analysis = df_analysis[df_analysis['Cluster'] != -1]
    
    cluster_stats = []
    for cluster in sorted(df_analysis['Cluster'].unique()):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        
        stats = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Avg Age': cluster_data['Age'].mean(),
            'Avg Purchase ($)': cluster_data['Purchase Amount (USD)'].mean(),
            'Avg Previous Purchases': cluster_data['Previous Purchases'].mean(),
            'Subscription Rate (%)': (cluster_data['Subscription Status'] == 'Yes').mean() * 100,
            'Discount Usage (%)': (cluster_data['Discount Applied'] == 'Yes').mean() * 100
        }
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🎯 Advanced Customer Segmentation Dashboard")
    st.markdown("**Interactive Analysis of Customer Segments using Unsupervised Machine Learning**")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        df, X, clustering_features = load_and_preprocess_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure 'shopping_trends.csv' is in the same directory.")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🔧 Clustering Configuration")
    
    algorithm = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        ['K-Means', 'DBSCAN', 'Gaussian Mixture']
    )
    
    if algorithm in ['K-Means', 'Gaussian Mixture']:
        n_clusters = st.sidebar.slider(
            "Number of Clusters",
            min_value=2, max_value=10, value=5
        )
    else:
        n_clusters = None
        eps = st.sidebar.slider(
            "DBSCAN Epsilon",
            min_value=0.1, max_value=2.0, value=0.5, step=0.1
        )
        min_samples = st.sidebar.slider(
            "DBSCAN Min Samples",
            min_value=2, max_value=20, value=5
        )
    
    # Run clustering
    if st.sidebar.button("🚀 Run Clustering Analysis"):
        with st.spinner(f"Running {algorithm} clustering..."):
            if algorithm == 'DBSCAN':
                labels, X_scaled, X_pca, sil_score, ch_score, db_score, explained_var = perform_clustering(
                    X, algorithm, eps=eps, min_samples=min_samples
                )
            else:
                labels, X_scaled, X_pca, sil_score, ch_score, db_score, explained_var = perform_clustering(
                    X, algorithm, n_clusters
                )
        
        # Store results in session state
        st.session_state.labels = labels
        st.session_state.X_pca = X_pca
        st.session_state.algorithm = algorithm
        st.session_state.metrics = {
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score,
            'explained_variance': explained_var
        }
    
    # Display results if clustering has been performed
    if hasattr(st.session_state, 'labels'):
        
        # Metrics row
        st.subheader("📊 Clustering Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Silhouette Score",
                f"{st.session_state.metrics['silhouette']:.3f}",
                help="Higher is better (-1 to 1)"
            )
        
        with col2:
            st.metric(
                "Calinski-Harabasz",
                f"{st.session_state.metrics['calinski_harabasz']:.1f}",
                help="Higher is better"
            )
        
        with col3:
            st.metric(
                "Davies-Bouldin",
                f"{st.session_state.metrics['davies_bouldin']:.3f}",
                help="Lower is better"
            )
        
        with col4:
            st.metric(
                "PCA Variance Explained",
                f"{st.session_state.metrics['explained_variance'].sum():.1%}",
                help="Variance captured by 2D visualization"
            )
        
        st.markdown("---")
        
        # Visualization
        st.subheader("🎨 Cluster Visualization")
        fig = create_cluster_visualization(
            st.session_state.X_pca, 
            st.session_state.labels, 
            st.session_state.algorithm
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster characteristics
        st.subheader("📋 Cluster Characteristics")
        cluster_df = create_cluster_characteristics(df, st.session_state.labels)
        st.dataframe(cluster_df, use_container_width=True)
        
        # Additional visualizations
        st.subheader("📈 Detailed Cluster Analysis")
        
        # Create detailed analysis plots
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = st.session_state.labels
        
        # Remove noise points for analysis
        if -1 in st.session_state.labels:
            df_with_clusters = df_with_clusters[df_with_clusters['Cluster'] != -1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Purchase amount by cluster
            fig_purchase = px.box(
                df_with_clusters, 
                x='Cluster', 
                y='Purchase Amount (USD)',
                title='Purchase Amount Distribution by Cluster',
                color='Cluster'
            )
            st.plotly_chart(fig_purchase, use_container_width=True)
        
        with col2:
            # Age distribution by cluster
            fig_age = px.histogram(
                df_with_clusters, 
                x='Age', 
                color='Cluster',
                title='Age Distribution by Cluster',
                marginal='box'
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Category preferences
        st.subheader("🛍️ Category Preferences by Cluster")
        category_cluster = df_with_clusters.groupby(['Cluster', 'Category']).size().reset_index(name='Count')
        fig_category = px.bar(
            category_cluster, 
            x='Category', 
            y='Count', 
            color='Cluster',
            title='Product Category Preferences by Cluster',
            barmode='group'
        )
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Business insights
        st.subheader("💡 Business Insights & Recommendations")
        
        insights_container = st.container()
        with insights_container:
            
            for cluster in sorted(df_with_clusters['Cluster'].unique()):
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
                cluster_size = len(cluster_data)
                avg_purchase = cluster_data['Purchase Amount (USD)'].mean()
                subscription_rate = (cluster_data['Subscription Status'] == 'Yes').mean()
                
                with st.expander(f"📊 Cluster {cluster} Analysis ({cluster_size} customers)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Purchase", f"${avg_purchase:.2f}")
                    with col2:
                        st.metric("Subscription Rate", f"{subscription_rate:.1%}")
                    with col3:
                        st.metric("Cluster Size", f"{cluster_size}")
                    
                    # Generate recommendations
                    if avg_purchase > 70 and subscription_rate > 0.8:
                        archetype = "Premium Loyal Customers"
                        recommendations = [
                            "🌟 Implement VIP loyalty programs",
                            "📧 Send personalized premium recommendations",
                            "🎁 Provide early access to new collections"
                        ]
                    elif subscription_rate < 0.3 and avg_purchase < 50:
                        archetype = "Price-Sensitive Shoppers"
                        recommendations = [
                            "💰 Target with discount campaigns",
                            "🔔 Send price drop notifications",
                            "📦 Offer bundle deals"
                        ]
                    else:
                        archetype = "Regular Value Customers"
                        recommendations = [
                            "🔄 Optimize subscription programs",
                            "📈 Focus on upselling strategies",
                            "🎪 Run seasonal campaigns"
                        ]
                    
                    st.write(f"**Segment Type:** {archetype}")
                    st.write("**Recommended Actions:**")
                    for rec in recommendations:
                        st.write(f"• {rec}")
    
    else:
        # Welcome message
        st.info("👆 Configure clustering parameters in the sidebar and click 'Run Clustering Analysis' to begin!")
        
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            st.metric("Features", len(clustering_features))
        with col3:
            st.metric("Avg Purchase Amount", f"${df['Purchase Amount (USD)'].mean():.2f}")
        
        # Show sample data
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Feature information
        st.subheader("🔧 Clustering Features")
        st.write("The following features will be used for clustering analysis:")
        
        feature_info = {
            'Age': 'Customer age',
            'Recency_Score': 'Customer engagement recency (Review Rating)',
            'Total_Purchase_Frequency': 'Combined purchase frequency score',
            'Monetary_Score': 'Purchase amount',
            'CLV_Proxy': 'Customer lifetime value estimate',
            'Is_Subscribed': 'Subscription status (binary)',
            'Uses_Discounts': 'Discount usage (binary)',
            'Uses_Promos': 'Promo code usage (binary)',
            'Previous_Purchases': 'Number of previous purchases',
            'Annual_Frequency': 'Annual purchase frequency',
            'Gender_Numeric': 'Gender (binary encoded)'
        }
        
        for feature, description in feature_info.items():
            st.write(f"• **{feature}**: {description}")

if __name__ == "__main__":
    main()
