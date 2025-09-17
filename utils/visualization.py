import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Dict, Optional

class OptimizationVisualizer:
    """
    Creates visualizations for optimization results
    """
    
    def __init__(self):
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2', 
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336',
            'info': '#2196F3'
        }
    
    def plot_optimization_progress(self, results: List[Dict], step_name: str) -> go.Figure:
        """
        Plot optimization progress showing fitness scores over iterations
        """
        if not results:
            return self._create_empty_plot("No optimization data available")
        
        # Extract fitness scores
        fitness_scores = [r.get('fitness', 0) for r in results]
        iterations = list(range(1, len(fitness_scores) + 1))
        
        # Create cumulative best scores
        cumulative_best = []
        best_so_far = 0
        for score in fitness_scores:
            if score > best_so_far:
                best_so_far = score
            cumulative_best.append(best_so_far)
        
        fig = go.Figure()
        
        # Add fitness scores
        fig.add_trace(go.Scatter(
            x=iterations,
            y=fitness_scores,
            mode='markers',
            name='Fitness Scores',
            marker=dict(
                color=self.color_palette['primary'],
                size=6,
                opacity=0.6
            ),
            hovertemplate='Iteration: %{x}<br>Fitness: %{y:.4f}<extra></extra>'
        ))
        
        # Add cumulative best line
        fig.add_trace(go.Scatter(
            x=iterations,
            y=cumulative_best,
            mode='lines',
            name='Best So Far',
            line=dict(
                color=self.color_palette['success'],
                width=3
            ),
            hovertemplate='Iteration: %{x}<br>Best Fitness: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{step_name} - Optimization Progress',
            xaxis_title='Iteration',
            yaxis_title='Fitness Score',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_parameter_distribution(self, results: List[Dict], parameter_name: str) -> go.Figure:
        """
        Plot distribution of a parameter across optimization results
        """
        if not results:
            return self._create_empty_plot("No parameter data available")
        
        # Extract parameter values and fitness scores
        param_values = []
        fitness_scores = []
        
        for result in results:
            params = result.get('parameters', {})
            if parameter_name in params:
                param_values.append(params[parameter_name])
                fitness_scores.append(result.get('fitness', 0))
        
        if not param_values:
            return self._create_empty_plot(f"No data for parameter: {parameter_name}")
        
        fig = go.Figure()
        
        # Create scatter plot
        fig.add_trace(go.Scatter(
            x=param_values,
            y=fitness_scores,
            mode='markers',
            name=f'{parameter_name} vs Fitness',
            marker=dict(
                color=fitness_scores,
                colorscale='Viridis',
                size=8,
                showscale=True,
                colorbar=dict(title="Fitness Score")
            ),
            hovertemplate=f'{parameter_name}: %{{x}}<br>Fitness: %{{y:.4f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Parameter Analysis: {parameter_name.replace("_", " ").title()}',
            xaxis_title=parameter_name.replace("_", " ").title(),
            yaxis_title='Fitness Score',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_parameter_correlation_heatmap(self, results: List[Dict]) -> go.Figure:
        """
        Create correlation heatmap between parameters and fitness
        """
        if not results:
            return self._create_empty_plot("No data for correlation analysis")
        
        # Extract all parameters into a DataFrame
        param_data = []
        for result in results:
            params = result.get('parameters', {}).copy()
            params['fitness'] = result.get('fitness', 0)
            param_data.append(params)
        
        if not param_data:
            return self._create_empty_plot("No parameter data available")
        
        df = pd.DataFrame(param_data)
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return self._create_empty_plot("Insufficient numeric parameters for correlation")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Parameter Correlation Matrix',
            template='plotly_white',
            height=500,
            width=500
        )
        
        return fig
    
    def plot_performance_metrics_comparison(self, results: List[Dict]) -> go.Figure:
        """
        Compare performance metrics across top results
        """
        if not results:
            return self._create_empty_plot("No results to compare")
        
        # Take top 10 results
        top_results = results[:10]
        
        metrics_to_plot = ['total_return', 'win_rate', 'profit_factor', 'max_drawdown', 'sharpe_ratio']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Total Return (%)', 'Win Rate (%)', 'Profit Factor', 
                           'Max Drawdown (%)', 'Sharpe Ratio', 'Fitness Score'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        ranks = list(range(1, len(top_results) + 1))
        
        # Plot each metric
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(positions):
                row, col = positions[i]
                values = [r.get('metrics', {}).get(metric, 0) for r in top_results]
                
                fig.add_trace(
                    go.Bar(
                        x=ranks,
                        y=values,
                        name=metric.replace('_', ' ').title(),
                        marker_color=self.color_palette['primary'],
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # Add fitness scores
        fitness_values = [r.get('fitness', 0) for r in top_results]
        fig.add_trace(
            go.Bar(
                x=ranks,
                y=fitness_values,
                name='Fitness Score',
                marker_color=self.color_palette['success'],
                showlegend=False
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title='Performance Metrics Comparison (Top 10 Results)',
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        # Update x-axis labels
        for i in range(1, 7):
            fig.update_xaxes(title_text="Rank", row=(i-1)//3+1, col=(i-1)%3+1)
        
        return fig
    
    def plot_equity_curve(self, trades: List[Dict], initial_balance: float = 10000) -> go.Figure:
        """
        Plot equity curve from trade data
        """
        if not trades:
            return self._create_empty_plot("No trade data available")
        
        # Calculate equity curve
        equity = [initial_balance]
        dates = [trades[0]['entry_time']]
        
        for trade in trades:
            equity.append(equity[-1] + trade.get('pnl', 0))
            dates.append(trade['exit_time'])
        
        # Calculate drawdown
        peak = initial_balance
        drawdown = []
        
        for balance in equity:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            drawdown.append(dd)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Equity Curve', 'Drawdown'],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='Equity',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='Date: %{x}<br>Balance: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color=self.color_palette['danger'], width=1),
                fill='tozeroy',
                fillcolor=f'rgba({self._hex_to_rgb(self.color_palette["danger"])}, 0.3)',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Backtest Results - Equity Curve and Drawdown',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Balance ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def plot_trade_analysis(self, trades: List[Dict]) -> go.Figure:
        """
        Analyze trade characteristics
        """
        if not trades:
            return self._create_empty_plot("No trade data available")
        
        trade_df = pd.DataFrame(trades)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['P&L Distribution', 'Trade Duration', 'Win/Loss by Time', 'Monthly Returns'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # P&L Distribution
        fig.add_trace(
            go.Histogram(
                x=trade_df['pnl'],
                nbinsx=20,
                name='P&L Distribution',
                marker_color=self.color_palette['primary'],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Trade Duration
        if 'trade_duration' in trade_df.columns:
            durations = trade_df['trade_duration'].dt.total_seconds() / 3600  # Convert to hours
            fig.add_trace(
                go.Histogram(
                    x=durations,
                    nbinsx=15,
                    name='Trade Duration',
                    marker_color=self.color_palette['secondary'],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Win/Loss by hour of day
        if 'entry_time' in trade_df.columns:
            trade_df['hour'] = pd.to_datetime(trade_df['entry_time']).dt.hour
            trade_df['is_winner'] = trade_df['pnl'] > 0
            
            hourly_stats = trade_df.groupby('hour')['is_winner'].agg(['count', 'sum']).reset_index()
            hourly_stats['win_rate'] = hourly_stats['sum'] / hourly_stats['count'] * 100
            
            fig.add_trace(
                go.Bar(
                    x=hourly_stats['hour'],
                    y=hourly_stats['win_rate'],
                    name='Win Rate by Hour',
                    marker_color=self.color_palette['success'],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Monthly returns
        if 'exit_time' in trade_df.columns:
            trade_df['month'] = pd.to_datetime(trade_df['exit_time']).dt.to_period('M')
            monthly_pnl = trade_df.groupby('month')['pnl'].sum()
            
            fig.add_trace(
                go.Bar(
                    x=[str(m) for m in monthly_pnl.index],
                    y=monthly_pnl.values,
                    name='Monthly P&L',
                    marker_color=self.color_palette['info'],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Trade Analysis Dashboard',
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_optimization_summary_table(self, results: List[Dict]) -> go.Figure:
        """
        Create a summary table of optimization results
        """
        if not results:
            return self._create_empty_plot("No results to display")
        
        # Prepare data for table
        table_data = []
        for i, result in enumerate(results[:10]):  # Top 10
            metrics = result.get('metrics', {})
            params = result.get('parameters', {})
            
            row = {
                'Rank': i + 1,
                'Fitness': f"{result.get('fitness', 0):.4f}",
                'Return (%)': f"{metrics.get('total_return', 0):.1f}",
                'Win Rate (%)': f"{metrics.get('win_rate', 0):.1f}",
                'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
                'Max DD (%)': f"{metrics.get('max_drawdown', 0):.1f}",
                'Trades': str(metrics.get('total_trades', 0)),
                'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color=self.color_palette['primary'],
                font=dict(color='white', size=12),
                align="center"
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color='white',
                font=dict(color='black', size=11),
                align="center",
                height=30
            )
        )])
        
        fig.update_layout(
            title='Optimization Results Summary',
            height=400,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """
        Create an empty plot with a message
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.color_palette['secondary'])
        )
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    def _hex_to_rgb(self, hex_color: str) -> str:
        """
        Convert hex color to RGB string
        """
        hex_color = hex_color.lstrip('#')
        return ','.join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))
    
    def display_optimization_visualizations(self, results: List[Dict], step_name: str):
        """
        Display all relevant visualizations for optimization results
        """
        if not results:
            st.warning("No optimization results to visualize")
            return
        
        # Progress plot
        st.plotly_chart(
            self.plot_optimization_progress(results, step_name),
            use_container_width=True
        )
        
        # Performance comparison
        st.plotly_chart(
            self.plot_performance_metrics_comparison(results),
            use_container_width=True
        )
        
        # Parameter analysis (if we have numeric parameters)
        numeric_params = self._get_numeric_parameters(results)
        if numeric_params:
            selected_param = st.selectbox(
                "Select parameter to analyze:",
                numeric_params,
                key=f"param_select_{step_name}"
            )
            
            st.plotly_chart(
                self.plot_parameter_distribution(results, selected_param),
                use_container_width=True
            )
        
        # Summary table
        st.plotly_chart(
            self.create_optimization_summary_table(results),
            use_container_width=True
        )
    
    def _get_numeric_parameters(self, results: List[Dict]) -> List[str]:
        """
        Get list of numeric parameters from results
        """
        if not results:
            return []
        
        numeric_params = []
        sample_params = results[0].get('parameters', {})
        
        for param, value in sample_params.items():
            if isinstance(value, (int, float)):
                numeric_params.append(param)
        
        return numeric_params
