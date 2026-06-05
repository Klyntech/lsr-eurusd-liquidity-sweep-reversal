"""
LSR-EH1 Strategy Configuration
All parameters frozen after IS2 validation (2019-2021).
No optimisation on holdout (2022-2024).
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SwingParams:
    lookback: int = 5          # bars on each side for swing detection (no look-ahead)
    swing_window: int = 20     # window to find most recent swing high/low
    atr_offset: float = 0.5    # ATR multiplier offset for liquidity level


@dataclass
class IndicatorParams:
    atr_period: int = 14       # ATR period (1h bars)
    adx_period: int = 14       # ADX period (4h bars)
    atr_percentile_window: int = 20  # rolling window for ATR percentile
    atr_percentile_min: int = 70     # minimum ATR percentile for trade


@dataclass
class FilterParams:
    adx_threshold: float = 25.0      # ADX threshold for trend exclusion
    min_bars_between_same_dir: int = 12  # 12 hours between same-direction entries


@dataclass
class ExitParams:
    profit_target_atr: float = 1.5  # take profit in ATR multiples
    stop_loss_atr: float = 1.0      # stop loss in ATR multiples
    max_bars_held: int = 6          # maximum holding period (6 hours)


@dataclass
class RiskParams:
    risk_per_trade: float = 0.01        # 1% equity risk per trade
    daily_loss_limit: float = 0.03       # 3% daily loss limit
    weekly_loss_limit: float = 0.05      # 5% weekly loss limit
    monthly_loss_limit: float = 0.08     # 8% monthly loss limit
    max_drawdown_hard: float = 0.15      # 15% hard stop from peak
    pip_value: float = 0.0001            # EURUSD pip value
    position_unit_step: int = 1000       # round down to nearest 1K units


@dataclass
class CostParams:
    half_spread_pips: float = 0.2        # round-trip half spread
    slippage_pips: float = 0.1           # market order slippage
    market_impact_pips: float = 0.05     # per $10M notional
    total_round_trip_pips: float = 0.35   # total cost in pips


@dataclass
class NewsBlackout:
    """Pre-scheduled high-impact news blackout windows."""
    name: str
    release_utc_hour: int
    blackout_minutes_before: int = 60
    blackout_minutes_after: int = 0

    def blackout_start_utc_hour(self) -> float:
        return self.release_utc_hour - self.blackout_minutes_before / 60.0

    def blackout_end_utc_hour(self) -> float:
        return self.release_utc_hour + self.blackout_minutes_after / 60.0


# Default news blacklist (fixed, not optimised)
NEWS_BLACKLIST: List[NewsBlackout] = [
    NewsBlackout(name="NFP", release_utc_hour=12, blackout_minutes_before=60, blackout_minutes_after=0),
    NewsBlackout(name="FOMC", release_utc_hour=18, blackout_minutes_before=60, blackout_minutes_after=0),
    NewsBlackout(name="ECB_Rate", release_utc_hour=12.75, blackout_minutes_before=60, blackout_minutes_after=0),
    NewsBlackout(name="ECB_PressConf", release_utc_hour=13.5, blackout_minutes_before=60, blackout_minutes_after=0),
    NewsBlackout(name="US_CPI", release_utc_hour=12.5, blackout_minutes_before=60, blackout_minutes_after=0),
]


@dataclass
class DataParams:
    instrument: str = "EURUSD"
    timeframe: str = "1H"
    spread_avg_pips: float = 0.2
    commission_per_million: float = 0.0   # FX spot, zero commission
    slippage_pips: float = 0.2             # assumed for market orders


@dataclass
class StrategyConfig:
    """Master configuration – single source of truth."""
    name: str = "LSR-EH1"
    swing: SwingParams = field(default_factory=SwingParams)
    indicator: IndicatorParams = field(default_factory=IndicatorParams)
    filter: FilterParams = field(default_factory=FilterParams)
    exit: ExitParams = field(default_factory=ExitParams)
    risk: RiskParams = field(default_factory=RiskParams)
    cost: CostParams = field(default_factory=CostParams)
    data: DataParams = field(default_factory=DataParams)
    news_blacklist: List[NewsBlackout] = field(default_factory=lambda: NEWS_BLACKLIST)
