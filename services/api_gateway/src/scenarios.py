from typing import Dict, List, Optional

SCENARIOS = {
    "scenario_1": {
        "id": "scenario_1",
        "name": "Cerah Berawan (Clear)",
        "description": "Kondisi normal, tidak ada hujan.",
        "data": {"rainfall_mm": 0.0, "water_level_cm": 50.0},
        "god_mode_enabled": False
    },
    "scenario_2": {
        "id": "scenario_2",
        "name": "Hujan Ringan (Light Rain)",
        "description": "Hujan intensitas rendah, aman.",
        "data": {"rainfall_mm": 10.5, "water_level_cm": 80.0},
        "god_mode_enabled": False
    },
    "scenario_3": {
        "id": "scenario_3",
        "name": "Banjir Historis (Historical)",
        "description": "Simulasi data banjir Jakarta 2020.",
        "data": {"rainfall_mm": 150.0, "water_level_cm": 220.0},
        "god_mode_enabled": False
    },
    "scenario_4": {
        "id": "scenario_4",
        "name": "Badai Ekstrem (Extreme)",
        "description": "Curah hujan sangat tinggi.",
        "data": {"rainfall_mm": 300.0, "water_level_cm": 280.0},
        "god_mode_enabled": False
    },
    "scenario_5": {
        "id": "scenario_5",
        "name": "GOD MODE (Force Critical)",
        "description": "Override sistem untuk demo darurat.",
        "data": {"rainfall_mm": 999.9, "water_level_cm": 999.9},
        "god_mode_enabled": True
    }
}

def get_scenario(scenario_id: str) -> Optional[Dict]:
    """Get a specific scenario by ID"""
    return SCENARIOS.get(scenario_id)

def list_scenarios() -> List[Dict]:
    """List all available scenarios for the frontend dropdown"""
    return list(SCENARIOS.values())
