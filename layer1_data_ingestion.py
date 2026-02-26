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