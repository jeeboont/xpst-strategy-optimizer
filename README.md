# XPST Strategy Optimizer

A comprehensive Streamlit application for optimizing XPST (eXtended Pivot SuperTrend) trading strategy parameters with integrated cTrader configuration file generation.

## Features

### 🎯 Strategy Optimization
- **3-Step Progressive Optimization**: Core parameters → Filters → Circuit Breaker & Re-Entry
- **Coarse-to-Fine Parameter Search**: Efficient exploration with precise fine-tuning
- **Multiple Timeframe Support**: 1m, 2m, 5m, 15m, 1h optimization
- **Comprehensive Backtesting**: Full strategy simulation with performance metrics

### 📊 Data Integration
- **YFinance Integration**: Automatic data download for any tradeable asset
- **Smart Asset Search**: Autocomplete search for stocks, crypto, forex, commodities
- **Data Validation**: Automatic quality checks and cleaning
- **Multiple Timeframe Downloads**: Bulk data export capability

### 🔧 cTrader Integration
- **cBot Configuration**: Ready-to-use .cbotset files
- **Indicator Configuration**: Ready-to-use .indiset files
- **Parameter Mapping**: Automatic translation of optimized parameters
- **Batch Export**: ZIP packages with documentation

### ⚡ Google Colab Integration
- **Hybrid Architecture**: Streamlit UI + Colab compute power
- **10x Faster Optimization**: Multi-core parallel processing
- **25GB RAM**: vs 1GB on Streamlit Cloud
- **GPU/TPU Support**: For advanced optimizations
- **Jupyter Notebook**: Ready-to-use optimization engine

### 📈 Advanced Analytics
- **Performance Metrics**: Return, win rate, profit factor, Sharpe ratio, drawdown
- **Interactive Visualizations**: Progress tracking, parameter analysis, equity curves
- **Trade Analysis**: Duration, P&L distribution, timing analysis
- **Risk Assessment**: Drawdown analysis, correlation studies

## Installation

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/xpst-strategy-optimizer.git
cd xpst-strategy-optimizer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repository
4. Deploy with default settings

## Usage Guide

### Step 1: Asset Selection
1. Use the search bar to find your desired asset
2. Search by symbol (e.g., "BTCUSD") or company name (e.g., "Bitcoin")
3. Select from autocomplete suggestions or use quick picks

### Step 2: Optimization Configuration
1. **Select Timeframe**: Choose from 1m to 1h timeframes
2. **Choose Optimization Steps**:
   - **Step 1 (Required)**: Core Parameters (Pivot Period, ATR Factor, ATR Period)
   - **Step 2 (Optional)**: Filters (XTrend, ADX, EMA)
   - **Step 3 (Optional)**: Circuit Breaker & Re-Entry
3. **Set Parameters**:
   - Data period (1 week to 3 months)
   - Risk per trade (0.5% to 5%)
   - Maximum combinations (100-1000)

### Step 3: Run Optimization
1. Click "Start Optimization"
2. Monitor progress in real-time
3. Review results as each step completes

### **Google Colab Optimization (Recommended for Complex Tasks)**

For intensive optimizations, use the Google Colab notebook:

1. **Access Colab Tab** in the Streamlit app
2. **Configure Parameters** using the interface
3. **Download Notebook** or click "Open in Google Colab"
4. **Run All Cells** in the notebook
5. **Download Results** and import to cTrader

**Performance Comparison:**
- **Streamlit Cloud**: 15-30 minutes (1 CPU, 1GB RAM)
- **Google Colab Free**: 3-5 minutes (2-4 CPUs, 12GB RAM)
- **Google Colab Pro**: 1-2 minutes (4-8 CPUs, 25GB RAM)

### **Local Streamlit Optimization**

For simpler optimizations or testing:

1. **Test the Full Workflow:**
   - Try optimizing a popular asset like BTCUSD or EURUSD
   - Run through all 3 optimization steps
   - Download the generated cTrader configuration files
   - Test the configurations in cTrader demo environment

## Optimization Strategy

### Step 1: Core Parameters
- **Pivot Period**: 2-15 (integer steps)
- **ATR Factor**: 0.8-2.0 (0.1 coarse, 0.05 fine-tuning)
- **ATR Period**: 10-40 (integer steps)

**Approach**: Coarse grid search (324 combinations) followed by fine-tuning around top performers

### Step 2: Filters
- **XTrend**: On/Off with MTF timeframe selection (m1, m2, m3, m5, m15, m30, h1)
- **ADX**: Threshold optimization (5-25)
- **EMA**: Period optimization (50-250, step 50)

**Approach**: Independent filter testing followed by best combination testing

### Step 3: Circuit Breaker & Re-Entry
- **CB Buffer**: 0.01-0.15% (0.01 steps)
- **Re-entry Cooldown**: 0-4 bars
- **Re-entry Window**: 0-20 bars

**Approach**: Systematic testing of CB settings, then re-entry optimization for XTrend strategies

## File Structure

```
xpst-strategy-optimizer/
├── app.py                          # Main Streamlit application
├── data_downloader.py              # YFinance data integration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── optimization/
│   ├── progressive_optimizer.py    # Main optimization controller
│   ├── core_optimizer.py          # Step 1: Core parameter optimization
│   ├── filter_optimizer.py        # Step 2: Filter optimization
│   ├── cb_reentry_optimizer.py    # Step 3: CB & re-entry optimization
│   └── backtesting_engine.py      # XPST strategy backtesting
└── utils/
    ├── config_generator.py        # cTrader configuration file generation
    ├── data_validation.py         # Data quality validation
    └── visualization.py           # Interactive charts and plots
```

## Technical Details

### Optimization Algorithm
- **Multi-objective fitness function**: Balances return, win rate, profit factor, drawdown
- **Early stopping**: Prevents over-optimization
- **Parallel processing**: Concurrent parameter testing
- **Smart sampling**: Intelligent parameter space exploration

### Data Requirements
- **Minimum bars**: 200-500 depending on timeframe
- **Quality validation**: Missing data, price integrity, volume checks
- **Automatic cleaning**: Invalid data removal and preparation

### Performance Metrics
- **Returns**: Total return percentage
- **Risk**: Maximum drawdown, Sharpe ratio, Sortino ratio
- **Trade Statistics**: Win rate, profit factor, average trade metrics
- **Risk-adjusted**: Calmar ratio, expectancy

## Configuration Files

### cBot Files (.cbotset)
- JSON format compatible with cTrader cBots
- Contains all optimized parameters
- Includes risk management and display settings
- Ready for direct import

### Indicator Files (.indiset)
- JSON format compatible with cTrader indicators
- String-formatted parameters
- Visual display configurations
- Chart overlay settings

## Important Disclaimers

### Risk Warnings
- **Historical Performance**: Past results don't guarantee future performance
- **Market Changes**: Optimization based on historical data may not suit future conditions
- **Demo Testing**: Always test thoroughly on demo accounts before live trading
- **Risk Management**: Use appropriate position sizing and risk controls

### Technical Limitations
- **Data Constraints**: Limited by YFinance data availability
- **Computational Limits**: Streamlit free tier has memory and processing constraints
- **Optimization Horizon**: Results based on specific historical periods
- **Market Regime**: Strategy performance may vary across different market conditions

## Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check the documentation
- Review the code comments for technical details

## License

This project is for educational and research purposes. Please ensure compliance with your broker's terms of service and local regulations when using automated trading strategies.

## Version History

### v3.0.0 (2025-09-17)
- Initial release with full optimization suite
- 3-step progressive optimization
- cTrader integration
- Interactive visualizations
- Comprehensive backtesting

---

**Disclaimer**: This software is provided for educational purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always consult with qualified financial professionals before making investment decisions.
