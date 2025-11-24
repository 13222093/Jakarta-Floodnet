#!/usr/bin/env python3
"""
Test script untuk memverifikasi komponen Dashboard
"""

import sys
from pathlib import Path

# Add components to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test semua imports berfungsi"""
    print("🧪 Testing imports...")
    
    try:
        from components import api_client
        print("  ✅ api_client imported successfully")
        
        from components import ui_widgets
        print("  ✅ ui_widgets imported successfully")
        
        import streamlit as st
        print("  ✅ streamlit imported successfully")
        
        import requests
        print("  ✅ requests imported successfully")
        
        import plotly
        print("  ✅ plotly imported successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {str(e)}")
        return False

def test_api_client():
    """Test API client functionality"""
    print("\n🧪 Testing API Client...")
    
    try:
        from components import api_client
        
        # Test health check
        print("  Testing health check...")
        health = api_client.check_health()
        
        if health['success']:
            print(f"  ✅ API Gateway online: {health['message']}")
        else:
            print(f"  ⚠️ API Gateway offline: {health['message']}")
        
        return True
    except Exception as e:
        print(f"  ❌ API Client test failed: {str(e)}")
        return False

def test_ui_widgets():
    """Test UI widgets"""
    print("\n🧪 Testing UI Widgets...")
    
    try:
        from components import ui_widgets
        
        # Test color constants
        assert hasattr(ui_widgets, 'COLORS'), "COLORS not found"
        print("  ✅ COLORS constant available")
        
        # Test functions exist
        functions = [
            'status_card',
            'metric_card',
            'error_message',
            'success_message',
            'warning_message',
            'connection_status_badge',
            'flood_level_indicator',
            'create_water_level_chart',
            'page_header'
        ]
        
        for func_name in functions:
            assert hasattr(ui_widgets, func_name), f"{func_name} not found"
            print(f"  ✅ {func_name} available")
        
        return True
    except Exception as e:
        print(f"  ❌ UI Widgets test failed: {str(e)}")
        return False

def test_pages_exist():
    """Test semua halaman ada"""
    print("\n🧪 Testing Pages...")
    
    pages = [
        "src/app.py",
        "src/pages/01_Dashboard_Utama.py",
        "src/pages/02_Prediksi_Banjir.py",
        "src/pages/03_Verifikasi_Visual.py"
    ]
    
    all_exist = True
    for page in pages:
        page_path = Path(__file__).parent / page
        if page_path.exists():
            print(f"  ✅ {page} exists")
        else:
            print(f"  ❌ {page} NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("=" * 60)
    print("🌊 Jakarta FloodNet Dashboard - Component Tests")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "API Client": test_api_client(),
        "UI Widgets": test_ui_widgets(),
        "Pages": test_pages_exist()
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Dashboard is ready to run.")
        print("\nTo start dashboard:")
        print("  cd services/dashboard/src")
        print("  streamlit run app.py")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before running.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
