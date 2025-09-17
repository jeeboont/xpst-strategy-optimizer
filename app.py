import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import zipfile
import io
import json
from datetime import datetime, timedelta
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import time

# Import custom modules
from data_downloader import YFinanceDataDownloader, TIMEFRAME_CONFIG
from optimization.progressive_optimizer import ProgressiveOptimizer
from optimization.backtesting_engine import XPSTBacktester
from utils.config_generator import ConfigurationGenerator
from utils.data_validation import DataValidator
from utils.visualization import OptimizationVisualizer

# App Configuration
APP_VERSION = "3.0.0"
VERSION_DATE = "2025-09-17"
CHANGELOG = {
    "3.0.0": {
        "date": "2025-09-17",
        "changes": [
            "Added XPST strategy optimization engine",
            "Integrated 3-step progressive optimization",
            "Added cBot and Indicator configuration file generation",
            "Enhanced data validation for optimization",
            "Added comprehensive visualization suite",
            "Implemented concurrent optimization processing"
        ]
    },
    "2.2.0": {
        "date": "2025-08-25",
        "changes": [
            "Added multiple timeframe selection capability",
            "Implemented automatic period limits based on yfinance constraints",
            "Added smart timeframe validation and recommendations",
            "Enhanced download progress tracking"
        ]
    }
}

# Set page config
st.set_page_config(
    page_title="XPST Strategy Optimizer", 
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'selected_assets': [],
        'selected_timeframes': ['2m'],
        'optimization_results': None,
        'current_optimization_step': 0,
        'show_search_results': False,
        'last_search_query': "",
        'search_results': [],
        'optimization_in_progress': False,
        'data_cache': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Custom CSS
def load_custom_css():
    """Load custom CSS for better UI"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        margin: 5px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .optimization-card {
        background: white;
        border: 2px solid #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .step-header {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .parameter-display {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .progress-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .download-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    init_session_state()
    load_custom_css()
    
    # Header
    header_html = f"""
    <div class="main-header">
        <h1>🎯 XPST Strategy Optimizer</h1>
        <p>Advanced Trading Strategy Optimization with cTrader Integration - v{APP_VERSION}</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Strategy Optimization", "📊 Data Download", "ℹ️ About"])
    
    with tab1:
        render_optimization_interface()
    
    with tab2:
        render_data_download_interface()
    
    with tab3:
        render_about_interface()

def render_optimization_interface():
    """Render the strategy optimization interface"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_optimization_sidebar()
    
    with col2:
        render_optimization_main()

def render_optimization_sidebar():
    """Render optimization configuration sidebar"""
    st.markdown("### 🔧 Optimization Configuration")
    
    # Asset Selection
    st.markdown("#### 📈 Select Asset")
    downloader = YFinanceDataDownloader()
    selected_asset = downloader.render_asset_selection_widget()
    
    if selected_asset:
        st.session_state.selected_assets = [selected_asset]
        st.success(f"Selected: {selected_asset}")
    
    # Timeframe Selection (limited to optimization-suitable timeframes)
    st.markdown("#### ⏰ Select Timeframe")
    optimization_timeframes = ['1m', '2m', '5m', '15m', '1h']
    
    selected_timeframe = st.selectbox(
        "Choose timeframe for optimization:",
        optimization_timeframes,
        index=1,  # Default to 2m
        help="Higher timeframes provide more reliable optimization results"
    )
    
    if selected_timeframe:
        st.session_state.selected_timeframes = [selected_timeframe]
    
    # Optimization Steps Selection
    st.markdown("#### 🎯 Optimization Steps")
    
    step1_enabled = st.checkbox("Step 1: Core Parameters", value=True, 
                               help="Optimize Pivot Period, ATR Factor, and ATR Period")
    step2_enabled = st.checkbox("Step 2: Filters", value=False,
                               help="Optimize XTrend, ADX, and EMA filters")
    step3_enabled = st.checkbox("Step 3: Circuit Breaker & Re-Entry", value=False,
                               help="Optimize circuit breaker and re-entry settings")
    
    optimization_steps = []
    if step1_enabled:
        optimization_steps.append("step1")
    if step2_enabled:
        optimization_steps.append("step2")
    if step3_enabled:
        optimization_steps.append("step3")
    
    # Data Period Selection
    st.markdown("#### 📅 Data Period")
    data_periods = {
        '1 week': '7d',
        '2 weeks': '14d', 
        '1 month': '1mo',
        '2 months': '60d',
        '3 months': '3mo'
    }
    
    selected_period_label = st.selectbox(
        "Historical data period:",
        list(data_periods.keys()),
        index=2,  # Default to 1 month
        help="More data provides better optimization but takes longer"
    )
    
    selected_period = data_periods[selected_period_label]
    
    # Risk Settings
    st.markdown("#### ⚠️ Risk Settings")
    risk_percent = st.slider(
        "Risk per trade (%):",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Percentage of account to risk per trade"
    )
    
    # Optimization Settings
    st.markdown("#### ⚙️ Optimization Settings")
    max_combinations = st.slider(
        "Max combinations to test:",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Higher values = more thorough but slower optimization"
    )
    
    # Start Optimization Button
    st.markdown("---")
    
    can_optimize = (
        len(st.session_state.selected_assets) > 0 and
        len(st.session_state.selected_timeframes) > 0 and
        len(optimization_steps) > 0 and
        not st.session_state.optimization_in_progress
    )
    
    if st.button(
        "🚀 Start Optimization",
        type="primary",
        use_container_width=True,
        disabled=not can_optimize
    ):
        # Store optimization parameters
        st.session_state.optimization_params = {
            'asset': st.session_state.selected_assets[0],
            'timeframe': st.session_state.selected_timeframes[0],
            'period': selected_period,
            'steps': optimization_steps,
            'risk_percent': risk_percent,
            'max_combinations': max_combinations
        }
        st.session_state.optimization_in_progress = True
        st.rerun()

def render_optimization_main():
    """Render main optimization content area"""
    
    if not st.session_state.optimization_in_progress and not st.session_state.optimization_results:
        render_optimization_instructions()
    
    elif st.session_state.optimization_in_progress:
        run_optimization_process()
    
    elif st.session_state.optimization_results:
        display_optimization_results()

def render_optimization_instructions():
    """Render optimization instructions when no optimization is running"""
    st.markdown("### 🎯 XPST Strategy Optimization")
    
    st.markdown("""
    Welcome to the XPST Strategy Optimizer! This tool helps you find the optimal parameters 
    for your trading strategy using historical data.
    
    #### 📋 How it works:
    
    **Step 1: Core Parameters** 🎯
    - Optimizes Pivot Point Period (2-15)
    - Optimizes ATR Factor (0.8-2.0) with fine-tuning
    - Optimizes ATR Period (10-40)
    
    **Step 2: Filters** 🔍 
    - Tests XTrend filter with different MTF timeframes
    - Optimizes ADX threshold (5-25)
    - Optimizes EMA period (50-250)
    
    **Step 3: Circuit Breaker & Re-Entry** ⚡
    - Optimizes Circuit Breaker buffer (0.01-0.15%)
    - Tests re-entry settings with cooldown and window periods
    
    #### 🎁 What you get:
    - Top 3 optimized parameter sets
    - Detailed performance metrics
    - Ready-to-use cBot (.cbotset) configuration files
    - Ready-to-use Indicator (.indiset) configuration files
    - Comprehensive backtesting reports
    
    👈 **Configure your optimization in the sidebar and click 'Start Optimization' to begin!**
    """)
    
    # Show example results
    with st.expander("📊 Example Results Preview"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">18.5%</div>
                <div class="metric-label">Total Return</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">67%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">1.8</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            """, unsafe_allow_html=True)

def run_optimization_process():
    """Execute the optimization process"""
    st.markdown("### 🚀 Optimization in Progress...")
    
    params = st.session_state.optimization_params
    
    try:
        # Step 1: Download and validate data
        with st.spinner(f"Downloading {params['asset']} data ({params['timeframe']})..."):
            downloader = YFinanceDataDownloader()
            data = downloader.download_optimization_data(
                params['asset'], 
                params['timeframe'], 
                params['period']
            )
            
            validator = DataValidator()
            validation = validator.validate_data_for_optimization(data, params['timeframe'])
            
            if not validation['is_valid']:
                st.error(f"Data validation failed: {validation['warnings']}")
                st.session_state.optimization_in_progress = False
                return
        
        st.success(f"✅ Data loaded: {len(data)} bars")
        
        # Step 2: Initialize optimizer
        optimizer = ProgressiveOptimizer(max_combinations=params['max_combinations'])
        
        # Step 3: Run optimization steps
        results = {}
        
        if 'step1' in params['steps']:
            with st.expander("🎯 Step 1: Core Parameters", expanded=True):
                progress_placeholder = st.empty()
                step1_results = optimizer.step1_core_optimization(
                    data, progress_callback=lambda p: progress_placeholder.progress(p)
                )
                results['step1'] = step1_results
                st.success(f"Step 1 completed! Found {len(step1_results)} candidates")
        
        if 'step2' in params['steps']:
            with st.expander("🔍 Step 2: Filter Optimization", expanded=True):
                if 'step1' in results:
                    progress_placeholder = st.empty()
                    step2_results = optimizer.step2_filter_optimization(
                        data, results['step1'], progress_callback=lambda p: progress_placeholder.progress(p)
                    )
                    results['step2'] = step2_results
                    st.success(f"Step 2 completed! Found {len(step2_results)} candidates")
                else:
                    st.warning("Step 1 must be completed before Step 2")
        
        if 'step3' in params['steps']:
            with st.expander("⚡ Step 3: CB & Re-Entry", expanded=True):
                base_results = results.get('step2', results.get('step1'))
                if base_results:
                    progress_placeholder = st.empty()
                    step3_results = optimizer.step3_cb_reentry_optimization(
                        data, base_results, progress_callback=lambda p: progress_placeholder.progress(p)
                    )
                    results['step3'] = step3_results
                    st.success(f"Step 3 completed! Found {len(step3_results)} candidates")
                else:
                    st.warning("Previous steps must be completed before Step 3")
        
        # Store results
        final_results = results.get('step3', results.get('step2', results.get('step1', [])))
        st.session_state.optimization_results = {
            'final_results': final_results[:3],  # Top 3
            'all_results': results,
            'asset': params['asset'],
            'timeframe': params['timeframe'],
            'data_stats': validation['data_stats']
        }
        
        st.session_state.optimization_in_progress = False
        st.success("🎉 Optimization completed successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        st.session_state.optimization_in_progress = False

def display_optimization_results():
    """Display optimization results with download options"""
    results = st.session_state.optimization_results
    
    st.markdown("### 🏆 Optimization Results")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Asset</div>
        </div>
        """.format(results['asset']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Timeframe</div>
        </div>
        """.format(results['timeframe']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Data Bars</div>
        </div>
        """.format(results['data_stats']['total_bars']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Candidates</div>
        </div>
        """.format(len(results['final_results'])), unsafe_allow_html=True)
    
    # Display top 3 results in tabs
    if len(results['final_results']) >= 1:
        tabs = st.tabs([f"🥇 Rank #{i+1}" for i in range(len(results['final_results'][:3]))])
        
        config_generator = ConfigurationGenerator()
        
        for i, (tab, result) in enumerate(zip(tabs, results['final_results'][:3])):
            with tab:
                display_single_result(result, results['asset'], results['timeframe'], i+1, config_generator)
    
    # Reset button
    if st.button("🔄 Run New Optimization", use_container_width=True):
        st.session_state.optimization_results = None
        st.session_state.optimization_in_progress = False
        st.rerun()

def display_single_result(result, asset, timeframe, rank, config_generator):
    """Display a single optimization result with download options"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Performance Metrics")
        metrics = result.get('metrics', {})
        
        # Create metrics display
        metrics_html = f"""
        <div class="parameter-display">
            <strong>Total Return:</strong> {metrics.get('total_return', 0):.2f}%<br>
            <strong>Win Rate:</strong> {metrics.get('win_rate', 0):.1f}%<br>
            <strong>Profit Factor:</strong> {metrics.get('profit_factor', 0):.2f}<br>
            <strong>Max Drawdown:</strong> {metrics.get('max_drawdown', 0):.2f}%<br>
            <strong>Total Trades:</strong> {metrics.get('total_trades', 0)}<br>
            <strong>Sharpe Ratio:</strong> {metrics.get('sharpe_ratio', 0):.2f}
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⚙️ Optimized Parameters")
        params = result.get('parameters', {})
        
        params_html = "<div class='parameter-display'>"
        for param, value in params.items():
            param_name = param.replace('_', ' ').title()
            params_html += f"<strong>{param_name}:</strong> {value}<br>"
        params_html += "</div>"
        
        st.markdown(params_html, unsafe_allow_html=True)
    
    # Configuration file downloads
    st.markdown("#### 📁 Download Configuration Files")
    
    col_cbot, col_indicator, col_zip = st.columns(3)
    
    # Generate configuration files
    cbot_config = config_generator.generate_cbot_config(params, asset, timeframe)
    indicator_config = config_generator.generate_indicator_config(params, asset, timeframe)
    
    with col_cbot:
        cbot_json = json.dumps(cbot_config, indent=2)
        st.download_button(
            label="📱 cBot Config",
            data=cbot_json,
            file_name=f"{asset}_{timeframe}_rank{rank}.cbotset",
            mime="application/json",
            use_container_width=True
        )
    
    with col_indicator:
        indicator_json = json.dumps(indicator_config, indent=2)
        st.download_button(
            label="📈 Indicator Config", 
            data=indicator_json,
            file_name=f"{asset}_{timeframe}_rank{rank}.indiset",
            mime="application/json",
            use_container_width=True
        )
    
    with col_zip:
        # Create ZIP with both files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"{asset}_{timeframe}_rank{rank}.cbotset", cbot_json)
            zip_file.writestr(f"{asset}_{timeframe}_rank{rank}.indiset", indicator_json)
            
            # Add README
            readme = f"""
XPST Optimization Result #{rank}
===============================
Asset: {asset}
Timeframe: {timeframe}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance:
- Total Return: {metrics.get('total_return', 0):.2f}%
- Win Rate: {metrics.get('win_rate', 0):.1f}%
- Profit Factor: {metrics.get('profit_factor', 0):.2f}

Import Instructions:
1. Import .cbotset in cTrader cBot section
2. Import .indiset in cTrader Indicators section
3. Apply to {asset} {timeframe} chart

Disclaimer: Test on demo before live trading.
"""
            zip_file.writestr("README.txt", readme)
        
        zip_buffer.seek(0)
        st.download_button(
            label="📦 Both as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{asset}_{timeframe}_rank{rank}_configs.zip",
            mime="application/zip",
            use_container_width=True
        )

def render_data_download_interface():
    """Render the data download interface (from your original script)"""
    st.markdown("### 📊 Data Download")
    st.info("Use this section to download historical data for manual analysis")
    
    downloader = YFinanceDataDownloader()
    downloader.render_full_interface()

def render_about_interface():
    """Render about/help interface"""
    st.markdown("### ℹ️ About XPST Strategy Optimizer")
    
    st.markdown(f"""
    **Version:** {APP_VERSION}  
    **Release Date:** {VERSION_DATE}
    
    This application provides comprehensive optimization for the XPST (eXtended Pivot SuperTrend) trading strategy.
    
    #### 🎯 Features:
    - **Progressive Optimization**: 3-step optimization process for maximum efficiency
    - **cTrader Integration**: Generate ready-to-use configuration files
    - **Data Validation**: Automatic data quality checks
    - **Performance Metrics**: Comprehensive backtesting analysis
    - **Multiple Timeframes**: Support for 1m to 1h timeframes
    
    #### 🔧 Technical Details:
    
    **Step 1: Core Parameters**
    - Pivot Point Period: 2-15 (integer steps)
    - ATR Factor: 0.8-2.0 (0.1 coarse, 0.05 fine-tuning)
    - ATR Period: 10-40 (integer steps)
    
    **Step 2: Filters**
    - XTrend: On/Off with MTF timeframe selection
    - ADX: Threshold optimization (5-25)
    - EMA: Period optimization (50-250, step 50)
    
    **Step 3: Circuit Breaker & Re-Entry**
    - CB Buffer: 0.01-0.15% (0.01 steps)
    - Re-entry Cooldown: 0-4 bars
    - Re-entry Window: 0-20 bars
    
    #### 📈 Optimization Algorithm:
    - **Coarse-to-Fine Search**: Efficient parameter space exploration
    - **Early Stopping**: Prevents over-optimization
    - **Concurrent Processing**: Faster optimization using multiple threads
    - **Smart Sampling**: Intelligent parameter combination selection
    
    #### ⚠️ Important Notes:
    - Results are based on historical data - past performance doesn't guarantee future results
    - Always test optimized parameters on demo accounts before live trading
    - Consider market regime changes when applying historical optimizations
    - Use appropriate position sizing and risk management
    
    #### 🔗 Resources:
    - Data Source: Yahoo Finance (yfinance)
    - Strategy: XPST (eXtended Pivot SuperTrend)
    - Platform: cTrader
    """)
    
    # Changelog
    with st.expander("📝 Changelog"):
        for version, info in CHANGELOG.items():
            st.markdown(f"**v{version}** - {info['date']}")
            for change in info['changes']:
                st.markdown(f"- {change}")
            st.markdown("---")

if __name__ == "__main__":
    main()
