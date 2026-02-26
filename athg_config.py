"""
ATHG Configuration Module
Centralized configuration for the Autonomous Trading Hypothesis Generator
"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    ccxt_exchanges: List[str] = None
    yfinance_tickers: List[str] = None
    alpaca_paper: bool = True
    alpaca_live: bool = False
    data_fetch_interval_minutes: int = 5
    max_historical_days: int = 365
    
    def __post_init__(self):
        if self.ccxt_exchanges is None:
            self.ccxt_exchanges = ["binance", "coinbase", "kraken"]
        if self.yfinance_tickers is None:
            self.yfinance_tickers = ["SPY", "QQQ", "BTC-USD", "ETH-USD"]

@dataclass
class FirebaseConfig:
    """Firebase configuration for state management"""
    project_id: Optional[str] = os.getenv("FIREBASE_PROJECT_ID")
    credentials_path: Optional[str] = os.getenv("FIREBASE_CREDENTIALS_PATH")
    collections: Dict[str, str] = None
    
    def __post_init__(self):
        if self.collections is None:
            self.collections = {
                "market_data": "market_data",
                "hypotheses": "trading_hypotheses",
                "anomalies": "market_anomalies",
                "backtests": "backtest_results",
                "performance": "strategy_performance"
            }

@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection"""
    zscore_threshold: float = 3.0
    isolation_forest_contamination: float = 0.1
    rolling_window_days: int = 30
    min_samples_for_analysis: int = 100
    correlation_threshold: float = 0.7

@dataclass
class LLMConfig:
    """LLM configuration for hypothesis generation"""
    model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 1000
    system_prompt_path: str = "prompts/system_prompt.txt"
    hypothesis_template_path: str = "prompts/hypothesis_template.txt"

@dataclass
class ATHGConfig:
    """Master configuration for ATHG"""
    data: DataSourceConfig = DataSourceConfig()
    firebase: FirebaseConfig = FirebaseConfig()
    anomalies: AnomalyDetectionConfig = AnomalyDetectionConfig()
    llm: LLMConfig = LLMConfig()
    logging_level: str = "INFO"
    max_hypotheses_per_cycle: int = 50
    backtest_lookback_days: int = 90
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings"""
        warnings = []
        
        # Firebase validation
        if not self.firebase.project_id:
            warnings.append("FIREBASE_PROJECT_ID not set in environment")
        if not self.firebase.credentials_path:
            warnings.append("FIREBASE_CREDENTIALS_PATH not set in environment")
        
        # Data source validation
        if len(self.data.yfinance_tickers) < 2:
            warnings.append("Less than 2 tickers configured for analysis")
        
        # Anomaly detection validation
        if self.anomalies.zscore_threshold < 2.0:
            warnings.append("Low z-score threshold may generate excessive anomalies")
        
        return warnings

# Global configuration instance
CONFIG = ATHGConfig()