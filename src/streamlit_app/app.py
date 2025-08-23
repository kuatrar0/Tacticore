#!/usr/bin/env python3
"""
Tacticore - CS2 Kill Event Labeling App

A Streamlit application for labeling Counter-Strike 2 kill events with
attacker and victim context, map overlays, and tactical analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from streamlit_app.components import (
    create_filters, apply_filters, create_label_controls, create_map_figure,
    display_kill_info, create_navigation_controls, display_labeled_summary,
    create_export_button, create_file_uploaders, create_map_settings
)
from streamlit_app.transforms import (
    load_map_data, get_kill_context
)

# Configure page
st.set_page_config(
    page_title="Tacticore - CS2 Kill Labeling",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kill-counter {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ff7f0e;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'current_kill_index' not in st.session_state:
        st.session_state.current_kill_index = 0
    
    if 'labeled_data' not in st.session_state:
        st.session_state.labeled_data = []
    
    if 'filtered_kills' not in st.session_state:
        st.session_state.filtered_kills = None


def save_labeled_kill(kill_context: Dict, attacker_label: str, victim_label: str) -> None:
    """
    Save a labeled kill to session state.
    
    Args:
        kill_context: Dictionary with kill context
        attacker_label: Label for attacker
        victim_label: Label for victim
    """
    labeled_kill = kill_context.copy()
    labeled_kill['attacker_label'] = attacker_label
    labeled_kill['victim_label'] = victim_label
    
    # Check if this kill is already labeled
    existing_indices = [
        i for i, labeled in enumerate(st.session_state.labeled_data)
        if (labeled.get('kill_tick') == kill_context.get('kill_tick') and
            labeled.get('attacker_name') == kill_context.get('attacker_name') and
            labeled.get('victim_name') == kill_context.get('victim_name'))
    ]
    
    if existing_indices:
        # Update existing label
        st.session_state.labeled_data[existing_indices[0]] = labeled_kill
        st.success("Updated existing label")
    else:
        # Add new label
        st.session_state.labeled_data.append(labeled_kill)
        st.success("Label saved!")


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Tacticore - CS2 Kill Labeling</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Tacticore")
    st.sidebar.markdown("Label CS2 kill events with tactical context")
    
    # File uploads
    kills_df, ticks_df, grenades_df = create_file_uploaders()
    
    # Check if required files are loaded
    if kills_df is None or ticks_df is None:
        st.warning("Please upload both kills.parquet and ticks.parquet files to continue.")
        return
    
    # Map settings
    map_image_path, map_data_path, tickrate, figure_size, x_fine_tune, y_fine_tune, use_advanced = create_map_settings()
    
    # Get selected map from session state
    selected_map = st.session_state.get("map_selection", "de_mirage")
    
    # Create filters
    filters = create_filters(kills_df)
    
    # Apply filters
    filtered_kills = apply_filters(kills_df, filters)
    st.session_state.filtered_kills = filtered_kills
    
    # Load map data after we have kills data
    map_data = {}
    if Path(map_data_path).exists():
        map_data = load_map_data(map_data_path, selected_map)
        

        
        # Override with manual adjustments if provided
        pos_x_adj = st.session_state.get("pos_x_adj", 0.0)
        pos_y_adj = st.session_state.get("pos_y_adj", 0.0)
        scale_adj = st.session_state.get("scale_adj", 5.0)
        
        if pos_x_adj != 0.0 or pos_y_adj != 0.0 or scale_adj != 5.0:
            map_data['pos_x'] = pos_x_adj
            map_data['pos_y'] = pos_y_adj
            map_data['scale'] = scale_adj
            st.success(f"Loaded map data for {selected_map} with manual adjustments")
        else:
            st.success(f"Loaded map data for {selected_map}: {map_data_path}")
        

    else:
        st.warning(f"Map data file not found: {map_data_path}")
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Kills:** {len(filtered_kills)} / {len(kills_df)}")
    
    if len(filtered_kills) == 0:
        st.warning("No kills match the current filters. Please adjust your filters.")
        return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Kill Analysis")
        
        # Kill counter
        current_index = st.session_state.current_kill_index
        total_kills = len(filtered_kills)
        
        st.markdown(f'<div class="kill-counter">Kill {current_index + 1} of {total_kills}</div>', 
                   unsafe_allow_html=True)
        
        # Get current kill
        current_kill = filtered_kills.iloc[current_index]
        
        # Get kill context
        rounds_df = pd.DataFrame()  # Empty for now, could be loaded if available
        kill_context = get_kill_context(
            current_kill, ticks_df, rounds_df, grenades_df if grenades_df is not None else pd.DataFrame(),
            map_data, tickrate, x_fine_tune, y_fine_tune, use_advanced
        )
        
        # Display kill information
        display_kill_info(kill_context)
        
        # Map visualization
        if map_data and Path(map_image_path).exists():
            st.subheader("🗺️ Map Location")
            
            attacker_pos = (kill_context['attacker_image_x'], kill_context['attacker_image_y'])
            victim_pos = (kill_context['victim_image_x'], kill_context['victim_image_y'])
            
            map_fig = create_map_figure(map_image_path, map_data, attacker_pos, victim_pos, figure_size)
            if map_fig:
                st.pyplot(map_fig)
        else:
            st.warning("Map image or data not available")
    
    with col2:
        st.header("🏷️ Labeling")
        
        # Label controls
        attacker_label, victim_label = create_label_controls()
        
        # Save button
        if st.button("💾 Save Labels", type="primary"):
            if attacker_label or victim_label:
                save_labeled_kill(kill_context, attacker_label, victim_label)
                
                # Auto-advance to next kill
                if current_index < total_kills - 1:
                    st.session_state.current_kill_index = current_index + 1
                    st.rerun()
            else:
                st.warning("Please select at least one label")
        
        # Navigation controls
        new_index = create_navigation_controls(total_kills, current_index)
        if new_index != current_index:
            st.session_state.current_kill_index = new_index
            st.rerun()
    
    # Bottom section
    st.markdown("---")
    
    # Labeled data summary
    if st.session_state.labeled_data:
        display_labeled_summary(st.session_state.labeled_data)
        
        # Export button
        create_export_button(st.session_state.labeled_data)
        
        # Clear all labels button
        if st.button("🗑️ Clear All Labels"):
            st.session_state.labeled_data = []
            st.rerun()
    
    # Instructions
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use Tacticore:
        
        1. **Upload Data**: Upload your parsed demo files (kills.parquet, ticks.parquet, optional grenades.parquet)
        2. **Configure Map**: Set the map image path and map data JSON file
        3. **Filter Kills**: Use the sidebar filters to focus on specific kills
        4. **Analyze Context**: Review the kill information and map location
        5. **Apply Labels**: Select appropriate labels for both attacker and victim
        6. **Save & Continue**: Save labels and automatically advance to the next kill
        7. **Export Results**: Download your labeled dataset when finished
        
        ### Labeling Guidelines:
        
        **Attacker Labels:**
        - `good_decision` - Tactically sound choice
        - `bad_decision` - Poor tactical choice  
        - `precise` - Accurate aim/execution
        - `imprecise` - Poor aim/execution
        - `good_positioning` - Good positioning
        - `bad_positioning` - Poor positioning
        - `other` - Other factors
        
        **Victim Labels:**
        - `exposed` - Victim was exposed
        - `no_cover` - Victim had no cover
        - `good_position` - Victim was well positioned
        - `mistake` - Victim made a mistake
        - `other` - Other factors
        """)


if __name__ == "__main__":
    main()
