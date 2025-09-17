import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import concurrent.futures

# Universal yfinance data limits
TIMEFRAME_CONFIG = {
    '1m': {
        'name': '1 Minute',
        'yf_interval': '1m',
        'max_days': 7,
        'recommended_period': '7d',
        'available_periods': ['1d', '2d', '5d', '7d'],
        'description': '1-minute data (max 7 days)',
        'icon': '🟢'
    },
    '2m': {
        'name': '2 Minutes',
        'yf_interval': '2m',
        'max_days': 60,
        'recommended_period': '1mo',
        'available_periods': ['1d', '2d', '5d', '7d', '1mo'],
        'description': '2-minute data (max 60 days)',
        'icon': '🟢'
    },
    '5m': {
        'name': '5 Minutes',
        'yf_interval': '5m',
        'max_days': 60,
        'recommended_period': '1mo',
        'available_periods': ['1d', '2d', '5d', '7d', '1mo'],
        'description': '5-minute data (max 60 days)',
        'icon': '🔵'
    },
    '15m': {
        'name': '15 Minutes',
        'yf_interval': '15m',
        'max_days': 60,
        'recommended_period': '1mo',
        'available_periods': ['1d', '2d', '5d', '7d', '1mo'],
        'description': '15-minute data (max 60 days)',
        'icon': '🔵'
    },
    '1h': {
        'name': '1 Hour',
        'yf_interval': '1h',
        'max_days': 60,
        'recommended_period': '1mo',
        'available_periods': ['1d', '5d', '1mo'],
        'description': 'Hourly data (max 60 days)',
        'icon': '🟡'
    },
    '4h': {
        'name': '4 Hours',
        'yf_interval': '1h',
        'max_days': 60,
        'recommended_period': '1mo',
        'available_periods': ['5d', '1mo'],
        'description': '4-hour data (resampled from 1h, max 60 days)',
        'icon': '🟠'
    },
    '1d': {
        'name': 'Daily',
        'yf_interval': '1d',
        'max_days': None,
        'recommended_period': '1y',
        'available_periods': ['5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
        'description': 'Daily data - extensive history available',
        'icon': '🔴'
    }
}

# Enhanced database for asset suggestions
POPULAR_ASSETS = {
    # Cryptocurrencies
    'bitcoin': {'symbol': 'BTC-USD', 'name': 'Bitcoin USD', 'sector': 'Cryptocurrency'},
    'btc': {'symbol': 'BTC-USD', 'name': 'Bitcoin USD', 'sector': 'Cryptocurrency'},
    'ethereum': {'symbol': 'ETH-USD', 'name': 'Ethereum USD', 'sector': 'Cryptocurrency'},
    'eth': {'symbol': 'ETH-USD', 'name': 'Ethereum USD', 'sector': 'Cryptocurrency'},
    'litecoin': {'symbol': 'LTC-USD', 'name': 'Litecoin USD', 'sector': 'Cryptocurrency'},
    'ltc': {'symbol': 'LTC-USD', 'name': 'Litecoin USD', 'sector': 'Cryptocurrency'},
    
    # Forex Pairs
    'eurusd': {'symbol': 'EURUSD=X', 'name': 'EUR/USD', 'sector': 'Forex'},
    'gbpusd': {'symbol': 'GBPUSD=X', 'name': 'GBP/USD', 'sector': 'Forex'},
    'usdjpy': {'symbol': 'USDJPY=X', 'name': 'USD/JPY', 'sector': 'Forex'},
    'audusd': {'symbol': 'AUDUSD=X', 'name': 'AUD/USD', 'sector': 'Forex'},
    'usdcad': {'symbol': 'USDCAD=X', 'name': 'USD/CAD', 'sector': 'Forex'},
    
    # Commodities
    'gold': {'symbol': 'GC=F', 'name': 'Gold Futures', 'sector': 'Commodity'},
    'silver': {'symbol': 'SI=F', 'name': 'Silver Futures', 'sector': 'Commodity'},
    'oil': {'symbol': 'CL=F', 'name': 'Crude Oil Futures', 'sector': 'Commodity'},
    'crude': {'symbol': 'CL=F', 'name': 'Crude Oil Futures', 'sector': 'Commodity'},
    
    # Major Stocks
    'apple': {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology'},
    'microsoft': {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'sector': 'Technology'},
    'google': {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'Technology'},
    'tesla': {'symbol': 'TSLA', 'name': 'Tesla, Inc.', 'sector': 'Automotive'},
    'amazon': {'symbol': 'AMZN', 'name': 'Amazon.com, Inc.', 'sector': 'E-commerce'},
    'nvidia': {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'sector': 'Technology'},
    
    # ETFs
    'spy': {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF Trust', 'sector': 'ETF'},
    'qqq': {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust', 'sector': 'ETF'}
}

class YFinanceDataDownloader:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for data downloader"""
        if 'selected_assets' not in st.session_state:
            st.session_state.selected_assets = []
        if 'show_search_results' not in st.session_state:
            st.session_state.show_search_results = False
        if 'last_search_query' not in st.session_state:
            st.session_state.last_search_query = ""
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
    
    @st.cache_data(ttl=300)
    def get_instant_suggestions(self, query):
        """Get instant suggestions from assets database with caching"""
        if not query or len(query) < 2:
            return []
        
        query_lower = query.lower()
        suggestions = []
        
        for key, asset in POPULAR_ASSETS.items():
            if (query_lower in key or 
                query_lower in asset['symbol'].lower() or 
                query_lower in asset['name'].lower()):
                suggestions.append(asset)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion['symbol'] not in seen:
                seen.add(suggestion['symbol'])
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= 8:
                    break
        
        return unique_suggestions
    
    @st.cache_data(ttl=300)
    def search_ticker(self, query):
        """Enhanced search for ticker symbols"""
        if not query or len(query) < 2:
            return []
        
        try:
            search_results = []
            query_upper = query.upper()
            
            # Create search variations
            variations = [
                query_upper,
                f"{query_upper}-USD",
                f"{query_upper}USD=X",
                f"USD{query_upper}=X",
                f"{query_upper}=F",
                f"{query_upper}.TO",
                f"{query_upper}.L"
            ]
            
            # Crypto mappings
            crypto_mappings = {
                'BITCOIN': 'BTC-USD',
                'ETHEREUM': 'ETH-USD',
                'LITECOIN': 'LTC-USD',
                'RIPPLE': 'XRP-USD'
            }
            
            if query_upper in crypto_mappings:
                variations.insert(0, crypto_mappings[query_upper])
            
            # Test variations
            for variation in variations[:5]:
                try:
                    ticker = yf.Ticker(variation)
                    info = ticker.info
                    
                    if info and ('longName' in info or 'shortName' in info):
                        name = info.get('longName') or info.get('shortName') or variation
                        if name and name != variation:
                            search_results.append({
                                'symbol': variation,
                                'name': name,
                                'sector': info.get('sector', 'Unknown'),
                                'exchange': info.get('exchange', 'Unknown')
                            })
                            if len(search_results) >= 3:
                                break
                except:
                    continue
            
            # Remove duplicates
            seen = set()
            unique_results = []
            for result in search_results:
                if result['symbol'] not in seen:
                    seen.add(result['symbol'])
                    unique_results.append(result)
            
            return unique_results[:3]
            
        except Exception:
            return []
    
    def render_asset_selection_widget(self) -> Optional[str]:
        """Render asset selection widget and return selected asset"""
        
        # Search input
        search_query = st.text_input(
            "Search for asset (symbol or name):",
            placeholder="e.g., BTCUSD, Apple, EURUSD...",
            key="asset_search"
        )
        
        # Auto-suggest as user types
        if search_query and search_query != st.session_state.last_search_query:
            st.session_state.last_search_query = search_query
            if len(search_query) >= 2:
                instant_suggestions = self.get_instant_suggestions(search_query)
                if instant_suggestions:
                    st.session_state.search_results = instant_suggestions
                    st.session_state.show_search_results = True
                else:
                    st.session_state.search_results = self.search_ticker(search_query)
                    st.session_state.show_search_results = True
            else:
                st.session_state.show_search_results = False
        
        # Display search results
        selected_asset = None
        
        if st.session_state.show_search_results and st.session_state.search_results:
            st.markdown("**Suggestions:**")
            
            for i, result in enumerate(st.session_state.search_results):
                col_info, col_select = st.columns([4, 1])
                
                with col_info:
                    st.markdown(f"**{result['symbol']}** - {result['name']}")
                    st.caption(f"Sector: {result.get('sector', 'N/A')}")
                
                with col_select:
                    if st.button("Select", key=f"select_{result['symbol']}_{i}"):
                        selected_asset = result['symbol']
                        st.session_state.show_search_results = False
                        st.session_state.last_search_query = ""
                        break
        
        elif st.session_state.show_search_results and not st.session_state.search_results and search_query:
            st.info("No suggestions found. Try a different term.")
        
        # Quick picks when no search
        if not search_query:
            st.markdown("**Quick Picks:**")
            popular_assets = [
                ('BTC-USD', 'Bitcoin'),
                ('ETH-USD', 'Ethereum'),
                ('EURUSD=X', 'EUR/USD'),
                ('GC=F', 'Gold'),
                ('AAPL', 'Apple'),
                ('TSLA', 'Tesla')
            ]
            
            cols = st.columns(3)
            for i, (symbol, name) in enumerate(popular_assets):
                with cols[i % 3]:
                    if st.button(f"{symbol}", key=f"quick_{symbol}", help=name, use_container_width=True):
                        selected_asset = symbol
                        break
        
        return selected_asset
    
    def download_optimization_data(self, symbol: str, timeframe: str, period: str) -> pd.DataFrame:
        """Download data specifically for optimization purposes"""
        try:
            config = TIMEFRAME_CONFIG[timeframe]
            ticker = yf.Ticker(symbol)
            
            # Download data
            data = ticker.history(
                period=period,
                interval=config['yf_interval'],
                auto_adjust=True,
                prepost=False
            )
            
            # Handle 4h resampling
            if timeframe == '4h' and not data.empty:
                data = self.resample_to_4h(data)
            
            # Validate minimum data requirements
            if len(data) < 50:
                raise ValueError(f"Insufficient data: only {len(data)} bars available")
            
            # Clean data
            data = data.dropna()
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            return data
            
        except Exception as e:
            raise Exception(f"Failed to download {symbol} data: {str(e)}")
    
    def resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-hour data to 4-hour data"""
        if df.empty:
            return df
        
        df_4h = df.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        return df_4h.dropna()
    
    def get_optimal_period_for_timeframe(self, timeframe: str) -> str:
        """Get optimal period for a given timeframe"""
        config = TIMEFRAME_CONFIG.get(timeframe, {})
        return config.get('recommended_period', '1mo')
    
    def download_single_asset_timeframe(self, asset: str, timeframe: str, period: str) -> Tuple[str, str, str, pd.DataFrame, str]:
        """Download data for a single asset and timeframe combination"""
        try:
            config = TIMEFRAME_CONFIG[timeframe]
            ticker = yf.Ticker(asset)
            
            data = ticker.history(
                period=period,
                interval=config['yf_interval']
            )
            
            if timeframe == '4h' and not data.empty:
                data = self.resample_to_4h(data)
            
            if not data.empty:
                return asset, timeframe, period, data, "success"
            else:
                return asset, timeframe, period, pd.DataFrame(), "no_data"
                
        except Exception as e:
            return asset, timeframe, period, pd.DataFrame(), f"error: {str(e)}"
    
    def render_full_interface(self):
        """Render the full data download interface from original script"""
        st.markdown("This feature allows you to download historical data for manual analysis.")
        st.info("For strategy optimization, use the main Optimization tab.")
        
        # Asset selection
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Select Assets:**")
            selected_asset = self.render_asset_selection_widget()
            
            if selected_asset and selected_asset not in st.session_state.selected_assets:
                st.session_state.selected_assets.append(selected_asset)
                st.rerun()
        
        with col2:
            st.markdown("**Selected Assets:**")
            if st.session_state.selected_assets:
                for asset in st.session_state.selected_assets:
                    col_asset, col_remove = st.columns([3, 1])
                    with col_asset:
                        st.text(asset)
                    with col_remove:
                        if st.button("Remove", key=f"remove_{asset}"):
                            st.session_state.selected_assets.remove(asset)
                            st.rerun()
            else:
                st.info("No assets selected")
        
        # Timeframe selection
        st.markdown("**Select Timeframes:**")
        timeframe_cols = st.columns(4)
        
        selected_timeframes = []
        for i, (tf_key, tf_config) in enumerate(TIMEFRAME_CONFIG.items()):
            with timeframe_cols[i % 4]:
                if st.checkbox(f"{tf_config['icon']} {tf_config['name']}", key=f"tf_download_{tf_key}"):
                    selected_timeframes.append(tf_key)
        
        # Download button
        if st.button("Download Data", type="primary", disabled=not (st.session_state.selected_assets and selected_timeframes)):
            self.execute_download(st.session_state.selected_assets, selected_timeframes)
    
    def execute_download(self, assets: List[str], timeframes: List[str]):
        """Execute the download process"""
        try:
            import zipfile
            import io
            
            total_combinations = len(assets) * len(timeframes)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            zip_buffer = io.BytesIO()
            successful_downloads = 0
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                current = 0
                
                for asset in assets:
                    for timeframe in timeframes:
                        current += 1
                        progress_bar.progress(current / total_combinations)
                        status_text.text(f'Downloading {asset} - {timeframe} ({current}/{total_combinations})')
                        
                        period = self.get_optimal_period_for_timeframe(timeframe)
                        asset_result, tf_result, period_result, data, status = self.download_single_asset_timeframe(
                            asset, timeframe, period
                        )
                        
                        if status == "success" and not data.empty:
                            filename = f"{asset}_{timeframe}_{period}_{datetime.now().strftime('%Y%m%d')}.csv"
                            csv_string = data.to_csv()
                            zip_file.writestr(filename, csv_string)
                            successful_downloads += 1
                            st.success(f"✅ {asset} - {timeframe}: {len(data)} records")
                        else:
                            st.error(f"❌ {asset} - {timeframe}: {status}")
            
            if successful_downloads > 0:
                zip_buffer.seek(0)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"yfinance_data_{timestamp}.zip"
                
                st.download_button(
                    label=f"📦 Download ZIP File ({successful_downloads} files)",
                    data=zip_buffer.getvalue(),
                    file_name=filename,
                    mime="application/zip",
                    use_container_width=True
                )
                
                st.success(f"Successfully downloaded {successful_downloads} files!")
            else:
                st.error("No data was successfully downloaded.")
                
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
