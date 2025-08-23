"""
Reusable Streamlit components for the labeling app.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple


def create_filters(kills_df: pd.DataFrame) -> Dict:
    """
    Create filter controls for kill events.
    
    Args:
        kills_df: DataFrame with kill data
        
    Returns:
        Dictionary with filter values
    """
    st.sidebar.header("Filters")
    
    filters = {}
    
    # Map filter - try both column names
    place_col = 'attacker_place' if 'attacker_place' in kills_df.columns else 'place'
    if place_col in kills_df.columns:
        places = ['All'] + sorted(kills_df[place_col].unique().tolist())
        filters['place'] = st.sidebar.selectbox("Map Place", places)
    
    # Side filter - try both column names
    side_col = 'attacker_side' if 'attacker_side' in kills_df.columns else 'side'
    if side_col in kills_df.columns:
        sides = ['All'] + sorted(kills_df[side_col].unique().tolist())
        filters['side'] = st.sidebar.selectbox("Side", sides)
    
    # Headshot filter
    if 'headshot' in kills_df.columns:
        headshot_options = ['All', 'Headshots Only', 'Non-headshots Only']
        filters['headshot'] = st.sidebar.selectbox("Headshot", headshot_options)
    
    # Time range filter
    if 'tick' in kills_df.columns:
        min_tick = int(kills_df['tick'].min())
        max_tick = int(kills_df['tick'].max())
        tick_range = st.sidebar.slider(
            "Tick Range", 
            min_value=min_tick, 
            max_value=max_tick,
            value=(min_tick, max_tick)
        )
        filters['tick_range'] = tick_range
    
    return filters


def apply_filters(kills_df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply filters to kills DataFrame.
    
    Args:
        kills_df: DataFrame with kill data
        filters: Dictionary with filter values
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = kills_df.copy()
    
    # Apply place filter - try both column names
    if 'place' in filters and filters['place'] != 'All':
        place_col = 'attacker_place' if 'attacker_place' in filtered_df.columns else 'place'
        if place_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[place_col] == filters['place']]
    
    # Apply side filter - try both column names
    if 'side' in filters and filters['side'] != 'All':
        side_col = 'attacker_side' if 'attacker_side' in filtered_df.columns else 'side'
        if side_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[side_col] == filters['side']]
    
    # Apply headshot filter
    if 'headshot' in filters:
        if filters['headshot'] == 'Headshots Only':
            filtered_df = filtered_df[filtered_df['headshot'] == True]
        elif filters['headshot'] == 'Non-headshots Only':
            filtered_df = filtered_df[filtered_df['headshot'] == False]
    
    # Apply tick range filter
    if 'tick_range' in filters:
        min_tick, max_tick = filters['tick_range']
        filtered_df = filtered_df[
            (filtered_df['tick'] >= min_tick) & 
            (filtered_df['tick'] <= max_tick)
        ]
    
    return filtered_df


def create_label_controls() -> Tuple[str, str]:
    """
    Create label selection controls.
    
    Returns:
        Tuple of (attacker_label, victim_label)
    """
    st.subheader("Label the Kill")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Attacker Label**")
        attacker_label = st.selectbox(
            "Attacker",
            ["", "good_decision", "bad_decision", "precise", "imprecise", 
             "good_positioning", "bad_positioning", "other"],
            key="attacker_label"
        )
    
    with col2:
        st.write("**Victim Label**")
        victim_label = st.selectbox(
            "Victim",
            ["", "exposed", "no_cover", "good_position", "mistake", "other"],
            key="victim_label"
        )
    
    return attacker_label, victim_label


def create_map_figure(map_image_path: str, map_data: Dict, 
                     attacker_pos: Tuple[float, float], 
                     victim_pos: Tuple[float, float],
                     figure_size: float = 5.0) -> plt.Figure:
    """
    Create map visualization with attacker and victim positions.
    
    Args:
        map_image_path: Path to map image
        map_data: Map coordinate data
        attacker_pos: (x, y) coordinates of attacker
        victim_pos: (x, y) coordinates of victim
        figure_size: Figure size in inches
        
    Returns:
        Matplotlib figure
    """
    try:
        # Load map image
        map_img = Image.open(map_image_path)
        img_width, img_height = map_img.size
        
        # Create figure with fixed aspect ratio to prevent size changes
        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        
        # Display map with fixed extent to prevent scaling issues
        ax.imshow(map_img, extent=[0, img_width, img_height, 0])
        ax.axis('off')
        
        # Set fixed limits to prevent auto-scaling
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Inverted Y axis for image coordinates
        
        # Clamp coordinates to image bounds
        def clamp_coords(x, y):
            return max(0, min(x, img_width - 1)), max(0, min(y, img_height - 1))
        
        attacker_x, attacker_y = clamp_coords(attacker_pos[0], attacker_pos[1])
        victim_x, victim_y = clamp_coords(victim_pos[0], victim_pos[1])
        
        # Plot attacker (blue)
        ax.scatter(attacker_x, attacker_y, 
                  c='blue', s=150, alpha=0.9, 
                  edgecolors='white', linewidth=3, label='Attacker', zorder=10)
        
        # Plot victim (red)
        ax.scatter(victim_x, victim_y, 
                  c='red', s=150, alpha=0.9,
                  edgecolors='white', linewidth=3, label='Victim', zorder=10)
        
        # Add labels with better visibility
        ax.annotate('Attacker', (attacker_x, attacker_y), 
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=10, color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='blue', alpha=0.8),
                   zorder=11)
        
        ax.annotate('Victim', (victim_x, victim_y),
                   xytext=(15, -15), textcoords='offset points',
                   fontsize=10, color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.8),
                   zorder=11)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Set title
        ax.set_title('Kill Location', fontsize=12, weight='bold', pad=10)
        
        # Ensure no padding/margins that could cause size changes
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating map figure: {e}")
        return None


def display_kill_info(context: Dict) -> None:
    """
    Display kill information in a formatted way.
    
    Args:
        context: Dictionary with kill context
    """
    st.subheader("Kill Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Info**")
        st.write(f"Attacker: {context.get('attacker_name', 'Unknown')}")
        st.write(f"Victim: {context.get('victim_name', 'Unknown')}")
        st.write(f"Attacker Side: {context.get('side', 'Unknown')}")
        st.write(f"Attacker Place: {context.get('place', 'Unknown')}")
        st.write(f"Headshot: {context.get('headshot', False)}")
    
    with col2:
        st.write("**Context**")
        st.write(f"Time in Round: {context.get('time_in_round_s', 0):.1f}s")
        st.write(f"Distance: {context.get('distance_xy', 0):.0f} units")
        st.write(f"Approach Alignment: {context.get('approach_align_deg', 0):.1f}°")
        st.write(f"Attacker Health: {context.get('attacker_health', 100)}")
        st.write(f"Victim Health: {context.get('victim_health', 100)}")
    
    # Utility flags
    st.write("**Nearby Utility**")
    utility_cols = st.columns(4)
    with utility_cols[0]:
        st.write(f"Flash: {'✓' if context.get('flash_near', False) else '✗'}")
    with utility_cols[1]:
        st.write(f"Smoke: {'✓' if context.get('smoke_near', False) else '✗'}")
    with utility_cols[2]:
        st.write(f"Molotov: {'✓' if context.get('molotov_near', False) else '✗'}")
    with utility_cols[3]:
        st.write(f"HE: {'✓' if context.get('he_near', False) else '✗'}")
    
    # Debug information (collapsible)
    if 'debug_attacker' in context or 'debug_victim' in context:
        with st.expander("🔧 Debug Coordinate Transformation"):
            st.write("**Coordinate Transformation Debug Info:**")
            
            if 'debug_attacker' in context:
                st.write("**Attacker Coordinates:**")
                debug = context['debug_attacker']
                st.write(f"Original: ({debug['original'][0]:.1f}, {debug['original'][1]:.1f})")
                st.write(f"After centering: ({debug['step1_centered'][0]:.1f}, {debug['step1_centered'][1]:.1f})")
                st.write(f"After scaling: ({debug['step2_scaled'][0]:.1f}, {debug['step2_scaled'][1]:.1f})")
                st.write(f"After centering offset: ({debug['step3_centered'][0]:.1f}, {debug['step3_centered'][1]:.1f})")
                st.write(f"After fine-tuning: ({debug['step4_adjusted'][0]:.1f}, {debug['step4_adjusted'][1]:.1f})")
                st.write(f"Final (clamped): ({debug['final'][0]:.1f}, {debug['final'][1]:.1f})")
            
            if 'debug_victim' in context:
                st.write("**Victim Coordinates:**")
                debug = context['debug_victim']
                st.write(f"Original: ({debug['original'][0]:.1f}, {debug['original'][1]:.1f})")
                st.write(f"After centering: ({debug['step1_centered'][0]:.1f}, {debug['step1_centered'][1]:.1f})")
                st.write(f"After scaling: ({debug['step2_scaled'][0]:.1f}, {debug['step2_scaled'][1]:.1f})")
                st.write(f"After centering offset: ({debug['step3_centered'][0]:.1f}, {debug['step3_centered'][1]:.1f})")
                st.write(f"After fine-tuning: ({debug['step4_adjusted'][0]:.1f}, {debug['step4_adjusted'][1]:.1f})")
                st.write(f"Final (clamped): ({debug['final'][0]:.1f}, {debug['final'][1]:.1f})")


def create_navigation_controls(total_kills: int, current_index: int) -> int:
    """
    Create navigation controls for moving between kills.
    
    Args:
        total_kills: Total number of kills
        current_index: Current kill index
        
    Returns:
        New kill index
    """
    st.subheader("Navigation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⏮️ First", key="first"):
            return 0
    
    with col2:
        if st.button("⬅️ Previous", key="prev"):
            return max(0, current_index - 1)
    
    with col3:
        if st.button("➡️ Next", key="next"):
            return min(total_kills - 1, current_index + 1)
    
    with col4:
        if st.button("⏭️ Last", key="last"):
            return total_kills - 1
    
    # Jump to specific index
    new_index = st.number_input(
        f"Jump to kill (0-{total_kills-1})", 
        min_value=0, 
        max_value=total_kills-1, 
        value=current_index,
        key="jump_to"
    )
    
    return new_index


def display_labeled_summary(labeled_data: List[Dict]) -> None:
    """
    Display summary of labeled kills.
    
    Args:
        labeled_data: List of labeled kill dictionaries
    """
    if not labeled_data:
        return
    
    st.subheader("Labeled Kills Summary")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(labeled_data)
    
    # Display counts
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Attacker Labels**")
        if 'attacker_label' in summary_df.columns:
            attacker_counts = summary_df['attacker_label'].value_counts()
            st.bar_chart(attacker_counts)
    
    with col2:
        st.write("**Victim Labels**")
        if 'victim_label' in summary_df.columns:
            victim_counts = summary_df['victim_label'].value_counts()
            st.bar_chart(victim_counts)
    
    # Display table
    st.write("**Labeled Data**")
    display_columns = ['attacker_name', 'victim_name', 'place', 'attacker_label', 'victim_label']
    available_columns = [col for col in display_columns if col in summary_df.columns]
    
    if available_columns:
        st.dataframe(summary_df[available_columns], use_container_width=True)


def create_export_button(labeled_data: List[Dict]) -> None:
    """
    Create export button for labeled data.
    
    Args:
        labeled_data: List of labeled kill dictionaries
    """
    if not labeled_data:
        st.warning("No labeled data to export")
        return
    
    st.subheader("Export Data")
    
    # Convert to DataFrame
    export_df = pd.DataFrame(labeled_data)
    
    # Create CSV
    csv_data = export_df.to_csv(index=False)
    
    # Download button
    st.download_button(
        label="📥 Download Labeled Data (CSV)",
        data=csv_data,
        file_name="features_labeled_context.csv",
        mime="text/csv"
    )
    
    st.success(f"Ready to export {len(labeled_data)} labeled kills")


def create_file_uploaders() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Create file upload widgets.
    
    Returns:
        Tuple of (kills_df, ticks_df, grenades_df)
    """
    st.header("Upload Data Files")
    
    # Kills file (required)
    kills_file = st.file_uploader(
        "Upload kills.parquet (required)", 
        type=['parquet'],
        key="kills_upload"
    )
    
    kills_df = None
    if kills_file is not None:
        try:
            kills_df = pd.read_parquet(kills_file)
            st.success(f"Loaded {len(kills_df)} kills")
        except Exception as e:
            st.error(f"Error loading kills file: {e}")
    
    # Ticks file (required)
    ticks_file = st.file_uploader(
        "Upload ticks.parquet (required)", 
        type=['parquet'],
        key="ticks_upload"
    )
    
    ticks_df = None
    if ticks_file is not None:
        try:
            ticks_df = pd.read_parquet(ticks_file)
            st.success(f"Loaded {len(ticks_df)} ticks")
        except Exception as e:
            st.error(f"Error loading ticks file: {e}")
    
    # Grenades file (optional)
    grenades_file = st.file_uploader(
        "Upload grenades.parquet (optional)", 
        type=['parquet'],
        key="grenades_upload"
    )
    
    grenades_df = None
    if grenades_file is not None:
        try:
            grenades_df = pd.read_parquet(grenades_file)
            st.success(f"Loaded {len(grenades_df)} grenade events")
        except Exception as e:
            st.error(f"Error loading grenades file: {e}")
    
    return kills_df, ticks_df, grenades_df


def create_map_settings() -> Tuple[str, str, int, float]:
    """
    Create map settings controls.
    
    Returns:
        Tuple of (map_image_path, map_data_path, tickrate, figure_size)
    """
    st.header("Map Settings")
    
    # Map selection
    available_maps = [
        "de_mirage", "de_dust2", "de_inferno", "de_nuke", 
        "de_overpass", "de_train", "de_vertigo", "de_cache",
        "de_cobblestone", "de_ancient", "de_anubis"
    ]
    
    selected_map = st.selectbox(
        "Select Map",
        available_maps,
        index=0,  # Default to de_mirage
        key="map_selection"
    )
    
    # Map image path
    map_image_path = st.text_input(
        "Map Image Path",
        value=f"maps/{selected_map}.png",
        key="map_image_path"
    )
    
    # Map data path
    map_data_path = st.text_input(
        "Map Data Path",
        value="maps/map-data.json",
        key="map_data_path"
    )
    
    # Coordinate transformation adjustments
    st.subheader("Coordinate Transformation")
    st.write("Adjust these values to fix player positions on the map:")
    
    pos_x = st.number_input(
        "Position X Offset",
        value=0.0,
        step=100.0,
        key="pos_x_adj"
    )
    
    pos_y = st.number_input(
        "Position Y Offset", 
        value=0.0,
        step=100.0,
        key="pos_y_adj"
    )
    
    scale = st.number_input(
        "Scale Factor",
        value=5.0,
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        key="scale_adj"
    )
    
    # Transformation method
    st.write("**Transformation Method:**")
    use_advanced = st.checkbox(
        "Use Advanced Transformation (Better for different map areas)",
        value=True,
        key="use_advanced_transform"
    )
    
    # Fine-tuning adjustments
    st.write("**Fine-tuning Adjustments:**")
    col1, col2 = st.columns(2)
    
    with col1:
        x_adjust = st.number_input(
            "X Fine-tune (±50)",
            value=25,
            min_value=-100,
            max_value=100,
            step=5,
            key="x_fine_tune"
        )
    
    with col2:
        y_adjust = st.number_input(
            "Y Fine-tune (±50)",
            value=0,
            min_value=-100,
            max_value=100,
            step=5,
            key="y_fine_tune"
        )
    
    # Tickrate
    tickrate = st.number_input(
        "Game Tickrate",
        min_value=32,
        max_value=128,
        value=64,
        step=32,
        key="tickrate"
    )
    
    # Figure size
    figure_size = st.slider(
        "Map Figure Size (inches)",
        min_value=3.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        key="figure_size"
    )
    
    return map_image_path, map_data_path, tickrate, figure_size, x_adjust, y_adjust, use_advanced
