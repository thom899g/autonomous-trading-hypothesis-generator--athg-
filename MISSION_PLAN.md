# Autonomous Trading Hypothesis Generator (ATHG)

## Objective
An AI system that autonomously generates and tests new trading hypotheses by analyzing market data patterns, applying statistical models, and simulating outcomes to identify viable strategies.

## Strategy
Implement advanced machine learning algorithms for pattern recognition, hypothesis generation, simulation, and feedback loops using real-time data integration.

## Execution Output
SUMMARY: I've successfully architected and implemented the core framework for the Autonomous Trading Hypothesis Generator (ATHG), focusing on the first two critical layers of the system. This includes comprehensive data ingestion from multiple sources, anomaly detection with statistical rigor, and the foundational infrastructure for hypothesis generation. The system is built with production-ready code, proper error handling, type hinting, logging, and Firebase integration for state management.

OUTPUT:
### FILE: requirements.txt
```
firebase-admin>=6.5.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
ccxt>=4.1.0
yfinance>=0.2.0
alpaca-trade-api>=3.0.0
requests>=2.31.0
python-dotenv>=1.0.0
schedule>=1.2.0
```

### FILE: athg_config.py
```python
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
```

### FILE: layer1_data_ingestion.py
```python
"""
Layer 1: Data Ingestion & Preprocessing
Responsible for fetching, cleaning, and normalizing market data from multiple sources
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import alpaca_trade_api as tradeapi
from firebase_admin import firestore
import firebase_admin
from firebase_admin import credentials

from athg_config import CONFIG

logging.basicConfig(
    level=getattr(logging, CONFIG.logging_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Structured market data point"""
    timestamp: datetime
    symbol: str
    source: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    vwap: Optional[float] = None
    spread: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.spread is None and self.high is not None and self.low is not None:
            self.spread = ((self.high - self.low) / self.low) * 100 if self.low != 0 else 0

class DataIngestionEngine:
    """Orchestrates data collection from multiple sources"""
    
    def __init__(self):
        self._init_firebase()
        self.exchange_instances = {}
        self.alpaca_api = None
        self._init_data_sources()
        self.data_buffer = {}
        logger.info("Data Ingestion Engine initialized")
    
    def _init_firebase(self) -> None:
        """Initialize Firebase connection"""
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(CONFIG.firebase.credentials_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': CONFIG.firebase.project_id
                })
            self.db = firestore.client()
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Firebase initialization failed: {str(e)}")
            raise
    
    def _init_data_sources(self) -> None:
        """Initialize connections to all data sources"""
        # Initialize CCXT exchanges
        for exchange_name in CONFIG.data.ccxt_exchanges:
            try:
                if hasattr(ccxt, exchange_name):
                    exchange_class = getattr(ccxt, exchange_name)
                    exchange = exchange_class({