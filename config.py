"""OmegaFinal-MLB configuration. Single source of truth for paths, teams, park factors."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent

# ── Park factors (Park Factor = runs scored at home / league average) ──
PARK_FACTORS: Dict[str, float] = {
    "Colorado Rockies": 1.34, "Boston Red Sox": 1.10, "Cincinnati Reds": 1.08,
    "Kansas City Royals": 1.06, "Chicago Cubs": 1.05, "Texas Rangers": 1.04,
    "Washington Nationals": 1.03, "Los Angeles Angels": 1.02, "Cleveland Guardians": 1.02,
    "Baltimore Orioles": 1.01, "Philadelphia Phillies": 1.01, "Arizona Diamondbacks": 1.01,
    "Chicago White Sox": 1.00, "Toronto Blue Jays": 1.00, "Atlanta Braves": 0.99,
    "Milwaukee Brewers": 0.99, "Minnesota Twins": 0.98, "Miami Marlins": 0.98,
    "Houston Astros": 0.98, "Detroit Tigers": 0.97, "Pittsburgh Pirates": 0.97,
    "St. Louis Cardinals": 0.97, "Oakland Athletics": 0.96, "Athletics": 0.96,
    "Los Angeles Dodgers": 0.96, "New York Yankees": 0.95, "New York Mets": 0.95,
    "San Francisco Giants": 0.95, "Tampa Bay Rays": 0.94, "San Diego Padres": 0.94,
    "Seattle Mariners": 0.92,
}

# ── Team MLB IDs (StatsAPI) ──
TEAM_IDS: Dict[str, int] = {
    "Arizona Diamondbacks": 109, "Atlanta Braves": 144, "Baltimore Orioles": 110,
    "Boston Red Sox": 111, "Chicago Cubs": 112, "Chicago White Sox": 145,
    "Cincinnati Reds": 113, "Cleveland Guardians": 114, "Colorado Rockies": 115,
    "Detroit Tigers": 116, "Houston Astros": 117, "Kansas City Royals": 118,
    "Los Angeles Angels": 108, "Los Angeles Dodgers": 119, "Miami Marlins": 146,
    "Milwaukee Brewers": 158, "Minnesota Twins": 142, "New York Mets": 121,
    "New York Yankees": 147, "Oakland Athletics": 133, "Athletics": 133,
    "Philadelphia Phillies": 143, "Pittsburgh Pirates": 134, "San Diego Padres": 135,
    "San Francisco Giants": 137, "Seattle Mariners": 136, "St. Louis Cardinals": 138,
    "Tampa Bay Rays": 139, "Texas Rangers": 140, "Toronto Blue Jays": 141,
    "Washington Nationals": 120,
}

# ── MLB divisions ──
DIVISIONS = {
    "AL East": ["Baltimore Orioles","Boston Red Sox","New York Yankees","Tampa Bay Rays","Toronto Blue Jays"],
    "AL Central": ["Chicago White Sox","Cleveland Guardians","Detroit Tigers","Kansas City Royals","Minnesota Twins"],
    "AL West": ["Houston Astros","Los Angeles Angels","Oakland Athletics","Athletics","Seattle Mariners","Texas Rangers"],
    "NL East": ["Atlanta Braves","Miami Marlins","New York Mets","Philadelphia Phillies","Washington Nationals"],
    "NL Central": ["Chicago Cubs","Cincinnati Reds","Milwaukee Brewers","Pittsburgh Pirates","St. Louis Cardinals"],
    "NL West": ["Arizona Diamondbacks","Colorado Rockies","Los Angeles Dodgers","San Diego Padres","San Francisco Giants"],
}

TEAM_DIV: Dict[str, str] = {}
for _div, _teams in DIVISIONS.items():
    for _t in _teams:
        TEAM_DIV[_t] = _div
TEAM_DIV['Athletics'] = TEAM_DIV.get('Oakland Athletics', 'AL West')


@dataclass(frozen=True)
class Paths:
    """File system paths for the OmegaFinal-MLB system."""
    base: Path = BASE_DIR
    db: Path = BASE_DIR / "data" / "omega_2026_BETA.db"
    db_backup: Path = BASE_DIR / "data" / "omega_2026_BETA.db.backup"
    trained: Path = BASE_DIR / "trained"
    snapshots: Path = BASE_DIR / "snapshots"
    logs: Path = BASE_DIR / "logs"

    @property
    def ensemble(self) -> Path: return self.trained / "ensemble"

    @property
    def k_props(self) -> Path: return self.trained / "k"

    @property
    def batter_props(self) -> Path: return self.trained / "props"

    @property
    def xwoba(self) -> Path: return self.trained / "xwoba"


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration settings: 2026 holdout for true out-of-sample calibration."""
    holdout_year: int = 2026
    min_bucket_size: int = 10
    bucket_size: int = 5
    meta_rf_boost_factor: float = 0.15  # max +/- 3 (was +/- 10, overconfident)


@dataclass(frozen=True)
class EnsembleConfig:
    """6-model ensemble × 4 time windows."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        'xgb1': 0.24, 'xgb2': 0.20, 'hgb1': 0.18, 'hgb2': 0.12, 'rf2': 0.06, 'mlp': 0.20,
    })
    windows: Dict[str, int] = field(default_factory=lambda: {
        'all': 0, 'w50': 750, 'w25': 375, 'w10': 150,
    })


@dataclass(frozen=True)
class Config:
    """Top-level config aggregating all sub-configs."""
    paths: Paths = field(default_factory=Paths)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    retrain_every_days: int = 1
    sync_recent_days: int = 2


CONFIG = Config()
