import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Game Optimization Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("summary.csv")

df = load_data()

# Title
st.title("🎮 Game Optimization Analysis Dashboard")
st.markdown("---")

# Sidebar for filters
st.sidebar.header("⚙️ Configuration")

# Get unique resolvers
resolvers = df['resolver'].unique().tolist()
selected_resolver = st.sidebar.selectbox("Resolver", resolvers)

st.sidebar.markdown("---")
st.sidebar.info("📊 The graphs below show how different parameter values affect game outcomes.")

# Filter data based on resolver only
filtered_df = df[df['resolver'] == selected_resolver]

# Main content area
if len(filtered_df) > 0:
    # Key metrics (overall averages)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg HP Lost", f"{filtered_df['hp_lost'].mean():.2f}")
    
    with col2:
        st.metric("Avg Enemies Killed", f"{filtered_df['killed_enemies'].mean():.2f}")
    
    with col3:
        st.metric("Victory Rate", f"{filtered_df['victory'].mean():.1%}")
    
    with col4:
        st.metric("Avg Rounds", f"{filtered_df['rounds'].mean():.2f}")
    
    st.markdown("---")
    
    # Define parameters to analyze
    parameters = {
        'playerMeleeDamage': 'Player Melee Damage',
        'playerShotDamage': 'Player Shot Damage',
        'playerHp': 'Player HP',
        'enemyMeleeDamage': 'Enemy Melee Damage',
        'enemyShotDamage': 'Enemy Shot Damage',
        'enemyHp': 'Enemy HP'
    }
    
    # Create tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["📉 HP Lost", "🎯 Enemies Killed", "🏆 Victory Rate", "⏱️ Rounds"])
    
    with tab1:
        st.subheader("HP Lost vs Parameters")
        for param, label in parameters.items():
            # Group by parameter and calculate mean
            grouped = filtered_df.groupby(param).agg({
                'hp_lost': ['mean', 'std', 'count']
            }).reset_index()
            grouped.columns = [param, 'mean', 'std', 'count']
            
            fig = px.line(
                grouped,
                x=param,
                y='mean',
                title=f"Average HP Lost vs {label}",
                labels={param: label, 'mean': 'Average HP Lost'},
                markers=True
            )
            
            # Add error bars for standard deviation
            fig.add_scatter(
                x=grouped[param],
                y=grouped['mean'] + grouped['std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
            fig.add_scatter(
                x=grouped[param],
                y=grouped['mean'] - grouped['std'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(68, 68, 68, 0.2)',
                showlegend=False,
                hoverinfo='skip'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Enemies Killed vs Parameters")
        for param, label in parameters.items():
            # Group by parameter and calculate mean
            grouped = filtered_df.groupby(param).agg({
                'killed_enemies': ['mean', 'std', 'count']
            }).reset_index()
            grouped.columns = [param, 'mean', 'std', 'count']
            
            fig = px.line(
                grouped,
                x=param,
                y='mean',
                title=f"Average Enemies Killed vs {label}",
                labels={param: label, 'mean': 'Average Enemies Killed'},
                markers=True,
                color_discrete_sequence=['#4ECDC4']
            )
            
            # Add error bars for standard deviation
            fig.add_scatter(
                x=grouped[param],
                y=grouped['mean'] + grouped['std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
            fig.add_scatter(
                x=grouped[param],
                y=grouped['mean'] - grouped['std'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(78, 205, 196, 0.2)',
                showlegend=False,
                hoverinfo='skip'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Victory Rate vs Parameters")
        for param, label in parameters.items():
            # Group by parameter and calculate victory rate
            grouped = filtered_df.groupby(param).agg({
                'victory': ['mean', 'count']
            }).reset_index()
            grouped.columns = [param, 'victory_rate', 'count']
            
            fig = px.line(
                grouped,
                x=param,
                y='victory_rate',
                title=f"Victory Rate vs {label}",
                labels={param: label, 'victory_rate': 'Victory Rate'},
                markers=True,
                color_discrete_sequence=['#95E1D3']
            )
            
            fig.update_yaxes(tickformat='.0%')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Number of Rounds vs Parameters")
        for param, label in parameters.items():
            # Group by parameter and calculate mean rounds
            grouped = filtered_df.groupby(param).agg({
                'rounds': ['mean', 'std', 'count']
            }).reset_index()
            grouped.columns = [param, 'mean', 'std', 'count']
            
            fig = px.line(
                grouped,
                x=param,
                y='mean',
                title=f"Average Rounds vs {label}",
                labels={param: label, 'mean': 'Average Rounds'},
                markers=True,
                color_discrete_sequence=['#F38181']
            )
            
            # Add error bars for standard deviation
            fig.add_scatter(
                x=grouped[param],
                y=grouped['mean'] + grouped['std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
            fig.add_scatter(
                x=grouped[param],
                y=grouped['mean'] - grouped['std'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(243, 129, 129, 0.2)',
                showlegend=False,
                hoverinfo='skip'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Show raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(filtered_df, use_container_width=True)

    
else:
    st.warning("⚠️ No data matches the selected parameters. Please adjust the filters.")
    st.info("💡 Try different combinations of parameters to see the results.")

# Footer
st.markdown("---")
st.markdown("*Dashboard created with Streamlit and Plotly*")
