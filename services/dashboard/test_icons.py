"""
Test file untuk verifikasi render_icon function
"""

import sys
sys.path.append('/run/media/reletz/Windows-SSD/Coding/Hackathon/Jakarta-Floodnet/services/dashboard/src')

from components.ui_widgets import render_icon

def test_render_icon():
    """Test basic icon rendering"""
    
    # Test basic icon
    waves_icon = render_icon('waves')
    print("✅ Waves icon generated:", waves_icon[:50] + "...")
    
    # Test with color
    alert_icon = render_icon('triangle-alert', size=32, color='#ef4444')
    print("✅ Alert icon with color generated:", alert_icon[:50] + "...")
    
    # Test brain circuit icon
    brain_icon = render_icon('brain-circuit', size=24, color='#3b82f6')
    print("✅ Brain circuit icon generated:", brain_icon[:50] + "...")
    
    # Test shield check icon  
    shield_icon = render_icon('shield-check', size=20, color='#10b981')
    print("✅ Shield check icon generated:", shield_icon[:50] + "...")
    
    print("\n🎉 All icon tests passed successfully!")

if __name__ == "__main__":
    test_render_icon()