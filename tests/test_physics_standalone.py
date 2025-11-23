"""
Standalone Physics Engine Logic Test
Demonstrates the physics constraints without FastAPI
"""

print("=" * 60)
print("🧪 Physics Engine Logic Demonstration")
print("=" * 60)

def apply_physics_engine(hujan_bogor, hujan_jakarta, tma_saat_ini, pred_lstm_raw=None):
    """
    Apply advanced physics engine logic
    """
    # If no LSTM prediction, use current level
    if pred_lstm_raw is None:
        pred_lstm_raw = tma_saat_ini
    
    # --- 1. Dynamic Weighting (Confidence-based) ---
    deviation = abs(pred_lstm_raw - tma_saat_ini)
    
    if deviation > 200: 
        ai_weight = 0.05
        physics_weight = 0.95
    elif deviation > 100:
        ai_weight = 0.15
        physics_weight = 0.85
    else:
        ai_weight = 0.30
        physics_weight = 0.70
    
    weighted_prediction = (tma_saat_ini * physics_weight) + (pred_lstm_raw * ai_weight)

    # --- 2. Proportional Rainfall Bias ---
    base_level = max(tma_saat_ini, 100)
    rain_bias = 0
    
    if hujan_bogor > 50 or hujan_jakarta > 50:
        rain_bias = base_level * 0.05  # +5% boost
    elif hujan_bogor > 20 or hujan_jakarta > 20:
        rain_bias = base_level * 0.02  # +2% boost
        
    rain_bias = min(rain_bias, 50.0)
    
    temp_pred = weighted_prediction + rain_bias

    # --- 3. SANITY CHECKS ---
    final_prediction = temp_pred
    
    # Rule A: Water Logic - If raining heavily, water CANNOT drop
    if (hujan_bogor > 20 or hujan_jakarta > 20):
        final_prediction = max(final_prediction, tma_saat_ini)
        
    # Rule B: Max Hourly Change
    max_change = 100.0
    if abs(final_prediction - tma_saat_ini) > max_change:
        if final_prediction > tma_saat_ini:
            final_prediction = tma_saat_ini + max_change
        else:
            final_prediction = tma_saat_ini - (max_change * 0.5)

    return round(max(final_prediction, 0.0), 2)

# Test Scenarios
scenarios = [
    {
        "name": "Heavy Rain + High Water (800cm)",
        "hujan_bogor": 150.0,
        "hujan_jakarta": 80.0,
        "tma_saat_ini": 800.0,
        "pred_lstm": 350.0,  # Model predicts drop (wrong!)
        "expected": ">= 800cm"
    },
    {
        "name": "Moderate Rain + Normal Water (400cm)",
        "hujan_bogor": 25.0,
        "hujan_jakarta": 20.0,
        "tma_saat_ini": 400.0,
        "pred_lstm": 380.0,
        "expected": ">= 400cm"
    },
    {
        "name": "No Rain + Low Water (300cm)",
        "hujan_bogor": 0.0,
        "hujan_jakarta": 0.0,
        "tma_saat_ini": 300.0,
        "pred_lstm": 290.0,
        "expected": "~300cm"
    },
    {
        "name": "Extreme Rain + Medium Water (500cm)",
        "hujan_bogor": 100.0,
        "hujan_jakarta": 60.0,
        "tma_saat_ini": 500.0,
        "pred_lstm": 400.0,  # Model predicts drop (wrong!)
        "expected": ">= 500cm"
    }
]

print("\n🚀 Running Tests...\n")

for i, scenario in enumerate(scenarios, 1):
    print(f"Test {i}: {scenario['name']}")
    print(f"  Input: Bogor={scenario['hujan_bogor']}mm, Jakarta={scenario['hujan_jakarta']}mm, TMA={scenario['tma_saat_ini']}cm")
    print(f"  LSTM Prediction (raw): {scenario['pred_lstm']}cm")
    
    result = apply_physics_engine(
        scenario['hujan_bogor'],
        scenario['hujan_jakarta'],
        scenario['tma_saat_ini'],
        scenario['pred_lstm']
    )
    
    delta = result - scenario['tma_saat_ini']
    heavy_rain = scenario['hujan_bogor'] > 20 or scenario['hujan_jakarta'] > 20
    
    # Check physics constraints
    if heavy_rain and result < scenario['tma_saat_ini']:
        status = "❌ FAILED - Water dropped during heavy rain!"
    elif heavy_rain and result >= scenario['tma_saat_ini']:
        status = "✅ PASSED - Water maintained/rose during rain"
    elif not heavy_rain and abs(delta) <= 50:
        status = "✅ PASSED - Reasonable change without rain"
    else:
        status = "⚠️ CHECK - Unusual behavior"
    
    print(f"  Final Prediction: {result}cm (Δ {delta:+.2f}cm)")
    print(f"  {status}")
    print(f"  Expected: {scenario['expected']}")
    print()

print("=" * 60)
print("✅ Physics Engine Logic Test Complete")
print("=" * 60)
print("\nKey Insights:")
print("1. Heavy rain prevents water level from dropping")
print("2. Rainfall adds proportional bias (2-5% of current level)")
print("3. Max hourly change capped at 100cm rise, 50cm drop")
print("4. AI predictions weighted based on deviation from reality")
