"""
Advanced Customer Segmentation Dashboard
Interactive Streamlit Application for Customer Segmentation Analysis

Author: [Your Name]
Course: BMCS2003 Artificial Intelligence
Assignment: Machine Learning (Unsupervised)

Enhanced Features:
- Customer cluster prediction tool
- Personalized recommendations
- Navigation controls
- Interactive feature selection
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

# Enhanced Custom CSS
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
    .customer-prediction {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .segment-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .navigation-button {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the shopping trends data with robust error handling"""
    try:
        # Try multiple ways to load the CSV file for cloud deployment
        df = None
        
        # Method 1: Direct loading
        try:
            df = pd.read_csv('shopping_trends.csv')
        except FileNotFoundError:
            # Method 2: Try different encodings and paths
            try:
                df = pd.read_csv('shopping_trends.csv', encoding='utf-8')
            except:
                try:
                    df = pd.read_csv('shopping_trends.csv', encoding='latin-1')
                except:
                    import os
                    # Method 3: Try absolute path
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    csv_path = os.path.join(current_dir, 'shopping_trends.csv')
                    df = pd.read_csv(csv_path)
        
        if df is None:
            raise FileNotFoundError("Could not load shopping_trends.csv")
        
        # Debug: Show actual columns in the CSV
        st.write("🔍 **DEBUG: CSV Columns Found:**")
        st.write(f"📊 **Total columns:** {len(df.columns)}")
        st.write(f"📋 **Column names:** {list(df.columns)}")
        st.write(f"📈 **Dataset shape:** {df.shape}")
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Verify required columns exist
        required_columns = ['Previous Purchases', 'Review Rating', 'Frequency of Purchases', 
                          'Purchase Amount (USD)', 'Subscription Status', 'Discount Applied', 
                          'Promo Code Used', 'Gender', 'Age']
        
        st.write("🔍 **DEBUG: Checking required columns:**")
        for col in required_columns:
            exists = col in df.columns
            st.write(f"  • {col}: {'✅' if exists else '❌'}")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.write(f"❌ **Missing columns:** {missing_columns}")
            st.write(f"📝 **Available columns:** {list(df.columns)}")
            raise KeyError(f"Missing required columns: {missing_columns}")
        
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
            'Previous_Purchases', 'Annual_Frequency', 'Gender_Numeric'
        ]
        
        # Debug: Check clustering features
        st.write("🔍 **DEBUG: Checking clustering features:**")
        for feature in clustering_features:
            exists = feature in df.columns
            st.write(f"  • {feature}: {'✅' if exists else '❌'}")
            if not exists:
                st.write(f"❌ **Feature '{feature}' not found in dataset**")
                st.write(f"📝 **Available columns:** {list(df.columns)}")
                raise KeyError(f"Feature '{feature}' not found in dataset")
        
        X = df[clustering_features].fillna(df[clustering_features].median())
        
        st.success(f"✅ **Data loaded successfully!**")
        st.write(f"📊 **Final dataset:** {len(df):,} rows, {len(clustering_features)} features")
        
        return df, X, clustering_features
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.write("🔍 **Troubleshooting Steps:**")
        st.write("1. Verify 'shopping_trends.csv' is in your GitHub repository root")
        st.write("2. Check the file was committed and pushed to GitHub")
        st.write("3. Ensure Streamlit Cloud can access the file")
        st.write("4. Try redeploying the app")
        
        # Debug information
        try:
            import os
            st.write(f"📁 Current directory: {os.getcwd()}")
            st.write(f"📄 Files found: {os.listdir('.')}")
        except:
            pass
            
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
    
    # Return model and scaler for prediction
    return labels, X_scaled, X_pca, sil_score, ch_score, db_score, pca.explained_variance_ratio_, model, scaler

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

def predict_customer_cluster(customer_features, model, scaler, clustering_features):
    """Predict which cluster a new customer belongs to"""
    
    # Create customer feature vector
    customer_df = pd.DataFrame([customer_features])
    customer_X = customer_df[clustering_features]
    
    # Scale the features
    customer_X_scaled = scaler.transform(customer_X)
    
    # Predict cluster
    cluster = model.predict(customer_X_scaled)[0]
    
    return cluster

def generate_segment_advice(cluster, df_with_clusters):
    """Generate personalized advice based on cluster assignment"""
    
    if cluster == -1:
        return {
            'segment_type': '🔍 Unique Customer Pattern',
            'characteristics': 'This customer has distinctive patterns that don\'t fit standard segments - potentially high-value prospect.',
            'recommendations': [
                '📞 Conduct personalized consultation to understand unique needs',
                '🎯 Create custom marketing approach',
                '💎 Assign dedicated customer success manager',
                '📊 Monitor closely for emerging segment patterns'
            ],
            'color': '#6c757d',
            'priority': 'High - Special Attention'
        }
    
    # Get cluster characteristics
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
    if len(cluster_data) == 0:
        return None
    
    avg_purchase = cluster_data['Purchase Amount (USD)'].mean()
    subscription_rate = (cluster_data['Subscription Status'] == 'Yes').mean()
    discount_usage = (cluster_data['Discount Applied'] == 'Yes').mean()
    avg_age = cluster_data['Age'].mean()
    top_category = cluster_data['Category'].mode()[0]
    
    # Determine segment type and advice
    if avg_purchase > 70 and subscription_rate > 0.8:
        return {
            'segment_type': '⭐ Premium Loyal Customer',
            'characteristics': f'High-value customers (avg ${avg_purchase:.0f}) with strong loyalty ({subscription_rate:.0%} subscription rate). Primary interest: {top_category}',
            'recommendations': [
                '🌟 Enroll in VIP loyalty program with exclusive perks',
                '📧 Send personalized premium product recommendations',
                '🎁 Offer early access to new collections and limited editions',
                '💎 Provide white-glove customer service experience',
                '🏆 Invite to exclusive member events and previews'
            ],
            'color': '#28a745',
            'priority': 'High - Retention Focus'
        }
    elif discount_usage > 0.8 and avg_purchase < 50:
        return {
            'segment_type': '💰 Price-Conscious Shopper',
            'characteristics': f'Budget-minded customers (avg ${avg_purchase:.0f}) who actively seek deals ({discount_usage:.0%} discount usage). Primary interest: {top_category}',
            'recommendations': [
                '🎯 Target with personalized discount campaigns',
                '🔔 Set up price drop alerts and flash sale notifications',
                '📦 Promote bundle deals and volume discounts',
                '💳 Offer loyalty points program for future savings',
                '⏰ Send time-limited promotional offers'
            ],
            'color': '#ffc107',
            'priority': 'Medium - Value Focus'
        }
    elif avg_age < 35 and top_category == 'Clothing':
        return {
            'segment_type': '👗 Fashion Enthusiast',
            'characteristics': f'Young trendsetters (avg {avg_age:.0f} years) passionate about style and fashion. Primary interest: {top_category}',
            'recommendations': [
                '📱 Engage through social media and influencer partnerships',
                '🎨 Showcase latest trends and seasonal collections',
                '📸 Encourage user-generated content and style sharing',
                '🎯 Optimize mobile shopping experience',
                '🌟 Create style guides and fashion inspiration content'
            ],
            'color': '#e83e8c',
            'priority': 'High - Engagement Focus'
        }
    elif avg_purchase > 50 and subscription_rate > 0.5:
        return {
            'segment_type': '🎯 Steady Value Customer',
            'characteristics': f'Reliable customers (avg ${avg_purchase:.0f}) with good loyalty ({subscription_rate:.0%} subscription rate). Primary interest: {top_category}',
            'recommendations': [
                '🔄 Optimize subscription programs and auto-renewals',
                '📈 Focus on cross-selling complementary products',
                '🎪 Run targeted seasonal campaigns',
                '💳 Offer flexible payment and delivery options',
                '📊 Send personalized product recommendations'
            ],
            'color': '#17a2b8',
            'priority': 'Medium - Growth Focus'
        }
    else:
        return {
            'segment_type': '📬 Re-engagement Opportunity',
            'characteristics': f'Occasional customers (avg ${avg_purchase:.0f}) with re-engagement potential. Primary interest: {top_category}',
            'recommendations': [
                '🎁 Send welcome back offers and incentives',
                '📬 Launch gentle re-engagement email campaigns',
                '🎯 Create targeted win-back promotions',
                '📊 Provide preference-based recommendations',
                '🔔 Set up soft reminder notifications'
            ],
            'color': '#6f42c1',
            'priority': 'Low - Recovery Focus'
        }

def customer_feature_input():
    """Create input widgets for customer features"""
    
    st.markdown("### 🎯 Customer Segment Predictor")
    st.markdown("**Enter customer characteristics to predict their segment and receive personalized recommendations:**")
    
    # Create input form
    with st.form("customer_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Demographics**")
            age = st.slider("Age", min_value=18, max_value=80, value=35, help="Customer's age")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
            
        with col2:
            st.markdown("**💰 Purchase Behavior**")
            purchase_amount = st.slider("Current Purchase Amount ($)", min_value=10, max_value=200, value=60, help="Amount of current purchase")
            previous_purchases = st.slider("Previous Purchases", min_value=0, max_value=50, value=15, help="Number of previous purchases")
            frequency = st.selectbox("Purchase Frequency", 
                                    ['Weekly', 'Bi-Weekly', 'Fortnightly', 'Monthly', 'Quarterly', 'Annually'],
                                    index=3, help="How often customer makes purchases")
            
        with col3:
            st.markdown("**⭐ Preferences & Engagement**")
            review_rating = st.slider("Review Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1, help="Average review rating given")
            subscription = st.selectbox("Subscription Status", ["Yes", "No"], help="Does customer have active subscription?")
            uses_discounts = st.selectbox("Uses Discounts", ["Yes", "No"], help="Does customer actively use discounts?")
            uses_promos = st.selectbox("Uses Promo Codes", ["Yes", "No"], help="Does customer use promotional codes?")
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Customer Segment", type="primary")
    
    if submitted:
        # Calculate derived features
        frequency_mapping = {
            'Weekly': 52, 'Bi-Weekly': 26, 'Fortnightly': 26, 
            'Monthly': 12, 'Quarterly': 4, 'Annually': 1
        }
        
        annual_frequency = frequency_mapping[frequency]
        
        customer_features = {
            'Age': age,
            'Recency_Score': review_rating,
            'Total_Purchase_Frequency': previous_purchases * annual_frequency,
            'Monetary_Score': purchase_amount,
            'CLV_Proxy': (purchase_amount * previous_purchases * annual_frequency) / 100,
            'Is_Subscribed': 1 if subscription == 'Yes' else 0,
            'Uses_Discounts': 1 if uses_discounts == 'Yes' else 0,
            'Uses_Promos': 1 if uses_promos == 'Yes' else 0,
            'Previous_Purchases': previous_purchases,
            'Annual_Frequency': annual_frequency,
            'Gender_Numeric': 1 if gender == 'Male' else 0
        }
        
        return customer_features
    
    return None

def display_prediction_results(customer_features, predicted_cluster, advice, df_with_clusters):
    """Display prediction results with comprehensive analysis"""
    
    # Main prediction result
    st.markdown("---")
    st.markdown(f"""
    <div class="customer-prediction">
        🎯 Predicted Customer Segment: Cluster {predicted_cluster}
        <br>
        {advice['segment_type']}
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Segment Analysis")
        st.success(f"**Priority Level:** {advice['priority']}")
        st.info(advice['characteristics'])
        
        # Customer profile summary
        st.markdown("**Customer Profile:**")
        st.write(f"• Age: {customer_features['Age']} years")
        st.write(f"• Purchase Amount: ${customer_features['Monetary_Score']}")
        st.write(f"• Previous Purchases: {customer_features['Previous_Purchases']}")
        st.write(f"• Subscription: {'Yes' if customer_features['Is_Subscribed'] else 'No'}")
        st.write(f"• Uses Discounts: {'Yes' if customer_features['Uses_Discounts'] else 'No'}")
    
    with col2:
        st.markdown("### 💡 Personalized Recommendations")
        for i, rec in enumerate(advice['recommendations'], 1):
            st.write(f"**{i}.** {rec}")
        
        # Action priority
        st.markdown("**Next Steps:**")
        if advice['priority'].startswith('High'):
            st.error("🚨 Immediate attention required - high-value segment")
        elif advice['priority'].startswith('Medium'):
            st.warning("⚠️ Moderate priority - growth opportunity")
        else:
            st.info("ℹ️ Lower priority - nurture when possible")
    
    # Cluster comparison
    if predicted_cluster != -1:
        st.markdown("### 📈 Benchmark Against Cluster")
        
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == predicted_cluster]
        
        metrics_cols = st.columns(4)
        
        with metrics_cols[0]:
            cluster_avg_age = cluster_data['Age'].mean()
            delta_age = customer_features['Age'] - cluster_avg_age
            st.metric(
                "Age vs Cluster Avg", 
                f"{customer_features['Age']} years",
                f"{delta_age:+.0f} years",
                delta_color="normal"
            )
        
        with metrics_cols[1]:
            cluster_avg_purchase = cluster_data['Purchase Amount (USD)'].mean()
            delta_purchase = customer_features['Monetary_Score'] - cluster_avg_purchase
            st.metric(
                "Purchase vs Cluster Avg", 
                f"${customer_features['Monetary_Score']}",
                f"${delta_purchase:+.0f}",
                delta_color="normal"
            )
        
        with metrics_cols[2]:
            cluster_avg_previous = cluster_data['Previous Purchases'].mean()
            delta_previous = customer_features['Previous_Purchases'] - cluster_avg_previous
            st.metric(
                "Previous Purchases vs Avg", 
                f"{customer_features['Previous_Purchases']}",
                f"{delta_previous:+.0f}",
                delta_color="normal"
            )
        
        with metrics_cols[3]:
            cluster_subscription_rate = (cluster_data['Subscription Status'] == 'Yes').mean()
            customer_subscription = "Yes" if customer_features['Is_Subscribed'] else "No"
            st.metric(
                "Subscription Status", 
                customer_subscription,
                f"Cluster: {cluster_subscription_rate:.0%}",
                delta_color="off"
            )
        
        # Similar customers
        st.markdown("### 👥 Similar Customers in This Segment")
        similar_customers = cluster_data.sample(min(5, len(cluster_data)))
        
        display_cols = ['Age', 'Gender', 'Purchase Amount (USD)', 'Previous Purchases', 
                       'Subscription Status', 'Category', 'Frequency of Purchases']
        st.dataframe(similar_customers[display_cols], use_container_width=True)

def main():
    """Main Streamlit application with enhanced navigation"""
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Header with navigation
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🎯 Advanced Customer Segmentation Dashboard")
        st.markdown("**Interactive Analysis of Customer Segments using Unsupervised Machine Learning**")
    
    with col2:
        # Navigation button when results are available
        if hasattr(st.session_state, 'labels'):
            if st.button("🏠 Return to Home", help="Start new analysis", type="secondary"):
                # Clear all session state
                keys_to_remove = ['labels', 'X_pca', 'algorithm', 'metrics', 'model', 'scaler']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.experimental_rerun()
    
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        df, X, clustering_features = load_and_preprocess_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure 'shopping_trends.csv' is in the same directory.")
        st.stop()
    
    # Main content based on analysis state
    if not hasattr(st.session_state, 'labels'):
        show_home_page(df, X, clustering_features)
    else:
        show_results_page(df, X, clustering_features)

def show_home_page(df, X, clustering_features):
    """Show home page with clustering configuration"""
    
    # Sidebar controls
    st.sidebar.header("🔧 Clustering Configuration")
    
    algorithm = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        ['K-Means', 'DBSCAN', 'Gaussian Mixture'],
        help="Choose the clustering algorithm for analysis"
    )
    
    if algorithm in ['K-Means', 'Gaussian Mixture']:
        n_clusters = st.sidebar.slider(
            "Number of Clusters",
            min_value=2, max_value=10, value=5,
            help="Number of customer segments to create"
        )
    else:
        n_clusters = None
        eps = st.sidebar.slider(
            "DBSCAN Epsilon",
            min_value=0.1, max_value=2.0, value=0.5, step=0.1,
            help="Maximum distance between samples in the same cluster"
        )
        min_samples = st.sidebar.slider(
            "DBSCAN Min Samples",
            min_value=2, max_value=20, value=5,
            help="Minimum number of samples in a cluster"
        )
    
    # Run clustering button
    if st.sidebar.button("🚀 Run Clustering Analysis", type="primary"):
        with st.spinner(f"Running {algorithm} clustering analysis..."):
            try:
                if algorithm == 'DBSCAN':
                    results = perform_clustering(X, algorithm, eps=eps, min_samples=min_samples)
                else:
                    results = perform_clustering(X, algorithm, n_clusters)
                
                labels, X_scaled, X_pca, sil_score, ch_score, db_score, explained_var, model, scaler = results
                
                # Store results in session state
                st.session_state.labels = labels
                st.session_state.X_pca = X_pca
                st.session_state.algorithm = algorithm
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.metrics = {
                    'silhouette': sil_score,
                    'calinski_harabasz': ch_score,
                    'davies_bouldin': db_score,
                    'explained_variance': explained_var
                }
                
                st.success(f"✅ {algorithm} clustering analysis completed successfully!")
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"❌ Error during clustering analysis: {str(e)}")
    
    # Welcome content
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
    
    # Sample data
    st.subheader("📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Feature information
    st.subheader("🔧 Clustering Features")
    st.write("The following features will be used for clustering analysis:")
    
    feature_info = {
        'Age': 'Customer age in years',
        'Recency_Score': 'Customer engagement recency (Review Rating proxy)',
        'Total_Purchase_Frequency': 'Combined purchase frequency score',
        'Monetary_Score': 'Purchase amount in USD',
        'CLV_Proxy': 'Customer lifetime value estimate',
        'Is_Subscribed': 'Subscription status (binary)',
        'Uses_Discounts': 'Discount usage behavior (binary)',
        'Uses_Promos': 'Promo code usage (binary)',
        'Previous_Purchases': 'Number of previous purchases',
        'Annual_Frequency': 'Estimated annual purchase frequency',
        'Gender_Numeric': 'Gender encoded as binary (1=Male, 0=Female)'
    }
    
    for feature, description in feature_info.items():
        st.write(f"• **{feature}**: {description}")

def show_results_page(df, X, clustering_features):
    """Show results page with analysis and prediction tool"""
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Clustering Results", "🎯 Customer Prediction", "💡 Business Insights"])
    
    with tab1:
        show_clustering_analysis(df)
    
    with tab2:
        show_customer_prediction_tool(df, clustering_features)
    
    with tab3:
        show_detailed_business_insights(df)

def show_clustering_analysis(df):
    """Display clustering analysis results"""
    
    # Performance metrics
    st.subheader("📊 Clustering Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Silhouette Score",
            f"{st.session_state.metrics['silhouette']:.3f}",
            help="Measures cluster cohesion and separation (-1 to 1, higher is better)"
        )
    
    with col2:
        st.metric(
            "Calinski-Harabasz",
            f"{st.session_state.metrics['calinski_harabasz']:.1f}",
            help="Variance ratio criterion (higher is better)"
        )
    
    with col3:
        st.metric(
            "Davies-Bouldin",
            f"{st.session_state.metrics['davies_bouldin']:.3f}",
            help="Average similarity measure (lower is better)"
        )
    
    with col4:
        st.metric(
            "PCA Variance Explained",
            f"{st.session_state.metrics['explained_variance'].sum():.1%}",
            help="Variance captured by 2D visualization"
        )
    
    st.markdown("---")
    
    # Cluster visualization
    st.subheader("🎨 Cluster Visualization")
    fig = create_cluster_visualization(
        st.session_state.X_pca, 
        st.session_state.labels, 
        st.session_state.algorithm
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("📋 Cluster Characteristics Summary")
    cluster_df = create_cluster_characteristics(df, st.session_state.labels)
    st.dataframe(cluster_df, use_container_width=True)

def show_customer_prediction_tool(df, clustering_features):
    """Show customer prediction tool with enhanced interface"""
    
    # Customer input
    customer_features = customer_feature_input()
    
    if customer_features is not None:
        if hasattr(st.session_state, 'model') and hasattr(st.session_state, 'scaler'):
            try:
                # Predict cluster
                predicted_cluster = predict_customer_cluster(
                    customer_features, 
                    st.session_state.model, 
                    st.session_state.scaler, 
                    clustering_features
                )
                
                # Prepare data for advice generation
                df_with_clusters = df.copy()
                df_with_clusters['Cluster'] = st.session_state.labels
                
                # Generate personalized advice
                advice = generate_segment_advice(predicted_cluster, df_with_clusters)
                
                if advice:
                    display_prediction_results(customer_features, predicted_cluster, advice, df_with_clusters)
                else:
                    st.error("Unable to generate recommendations for this cluster.")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
        else:
            st.error("⚠️ Clustering model not available. Please run clustering analysis first!")

def show_detailed_business_insights(df):
    """Show detailed business insights and recommendations"""
    
    st.subheader("💡 Comprehensive Business Insights")
    
    # Create analysis dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = st.session_state.labels
    
    # Remove noise points
    if -1 in st.session_state.labels:
        df_analysis = df_with_clusters[df_with_clusters['Cluster'] != -1]
        noise_count = (st.session_state.labels == -1).sum()
        st.warning(f"⚠️ {noise_count} customers identified as outliers (noise points) - these may represent unique high-value prospects")
    else:
        df_analysis = df_with_clusters
    
    # Overall insights
    total_clusters = len(df_analysis['Cluster'].unique())
    st.info(f"📈 Successfully identified {total_clusters} distinct customer segments from {len(df):,} customers")
    
    # Detailed cluster analysis
    for cluster in sorted(df_analysis['Cluster'].unique()):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        cluster_size = len(cluster_data)
        avg_purchase = cluster_data['Purchase Amount (USD)'].mean()
        subscription_rate = (cluster_data['Subscription Status'] == 'Yes').mean()
        
        with st.expander(f"📊 Cluster {cluster} Deep Dive ({cluster_size} customers, {cluster_size/len(df_analysis)*100:.1f}%)"):
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Purchase", f"${avg_purchase:.2f}")
            with col2:
                st.metric("Subscription Rate", f"{subscription_rate:.1%}")
            with col3:
                avg_age = cluster_data['Age'].mean()
                st.metric("Average Age", f"{avg_age:.0f} years")
            with col4:
                avg_previous = cluster_data['Previous Purchases'].mean()
                st.metric("Avg Previous Purchases", f"{avg_previous:.0f}")
            
            # Generate segment-specific insights
            advice = generate_segment_advice(cluster, df_with_clusters)
            
            if advice:
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.markdown(f"**Segment Type:** {advice['segment_type']}")
                    st.markdown(f"**Priority:** {advice['priority']}")
                    st.write(advice['characteristics'])
                
                with col_right:
                    st.markdown("**Recommended Actions:**")
                    for rec in advice['recommendations']:
                        st.write(f"• {rec}")
            
            # Additional cluster insights
            st.markdown("**Top Preferences:**")
            col_pref1, col_pref2, col_pref3 = st.columns(3)
            
            with col_pref1:
                top_category = cluster_data['Category'].value_counts().head(3)
                st.write("🛍️ **Categories:**")
                for cat, count in top_category.items():
                    st.write(f"• {cat}: {count}")
            
            with col_pref2:
                top_season = cluster_data['Season'].value_counts().head(3)
                st.write("🌍 **Seasons:**")
                for season, count in top_season.items():
                    st.write(f"• {season}: {count}")
            
            with col_pref3:
                top_freq = cluster_data['Frequency of Purchases'].value_counts().head(3)
                st.write("⏰ **Frequency:**")
                for freq, count in top_freq.items():
                    st.write(f"• {freq}: {count}")

if __name__ == "__main__":
    main()