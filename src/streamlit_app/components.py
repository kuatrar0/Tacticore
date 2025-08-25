"""
Streamlit UI components for the Tacticore labeling app.
Enhanced version with batch labeling and ML training features.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import base64


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


def create_label_controls() -> Tuple[List[str], List[str]]:
    """
    Create label selection controls with multiple selection support.
    
    Returns:
        Tuple of (attacker_labels, victim_labels) as lists
    """
    st.subheader("Label the Kill")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Attacker Labels** (Multiple selection)")
        attacker_labels = st.multiselect(
            "Attacker",
            ["good_decision", "bad_decision", "precise", "imprecise", 
             "good_positioning", "bad_positioning", "other"],
            key="attacker_labels"
        )
    
    with col2:
        st.write("**Victim Labels** (Multiple selection)")
        victim_labels = st.multiselect(
            "Victim",
            ["exposed", "no_cover", "good_position", "mistake", "bad_clearing", "other"],
            key="victim_labels"
        )
    
    return attacker_labels, victim_labels


def create_map_figure(map_image_path: str, map_data: Dict, 
                     attacker_pos: Tuple[float, float], 
                     victim_pos: Tuple[float, float],
                     figure_size: float = 5.0, 
                     attacker_name: str = "Attacker",
                     victim_name: str = "Victim") -> plt.Figure:
    """
    Create map visualization with attacker and victim positions.
    
    Args:
        map_image_path: Path to map image
        map_data: Map coordinate data
        attacker_pos: (x, y) coordinates of attacker
        victim_pos: (x, y) coordinates of victim
        figure_size: Figure size in inches
        attacker_name: Name of the attacker
        victim_name: Name of the victim
        
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
                  edgecolors='white', linewidth=3, label=attacker_name, zorder=10)
        
        # Plot victim (red)
        ax.scatter(victim_x, victim_y, 
                  c='red', s=150, alpha=0.9,
                  edgecolors='white', linewidth=3, label=victim_name, zorder=10)
        
        # Add smaller labels with player names
        ax.annotate(attacker_name, (attacker_x, attacker_y), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.8),
                   zorder=11)
        
        ax.annotate(victim_name, (victim_x, victim_y),
                   xytext=(10, -10), textcoords='offset points',
                   fontsize=8, color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
                   zorder=11)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        # Set title
        ax.set_title('Kill Location', fontsize=10, weight='bold', pad=10)
        
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
        approach_align = context.get('approach_align_deg')
        if approach_align is not None:
            st.write(f"Approach Alignment: {approach_align:.1f}°")
        else:
            st.write("Approach Alignment: Not moving")
        st.write(f"Attacker Health: {context.get('attacker_health', 100)}")
        st.write(f"Victim Health: {context.get('victim_health', 100)}")
        
        # Victim awareness
        if 'victim_was_aware' in context:
            st.write("**Victim Awareness:**")
            st.write(f"Was Aware: {'Yes' if context['victim_was_aware'] else 'No'}")
            st.write(f"Was Watching: {'Yes' if context.get('victim_was_watching', False) else 'No'}")
            st.write(f"Was Backstabbed: {'Yes' if context.get('victim_was_backstabbed', False) else 'No'}")
            confidence = context.get('awareness_confidence', 0)
            st.write(f"Confidence: {confidence:.1%}")
    
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


def create_labeled_data_importer() -> None:
    """
    Create widget to import existing labeled CSV data.
    """
    st.sidebar.markdown("### 📥 Import Existing Labels")
    
    labeled_csv = st.sidebar.file_uploader(
        "Upload existing labeled data (features_labeled_context.csv)",
        type=['csv'],
        key="labeled_csv_upload"
    )
    
    if labeled_csv is not None:
        try:
            # Read the CSV
            labeled_df = pd.read_csv(labeled_csv)
            st.sidebar.success(f"Loaded {len(labeled_df)} labeled kills")
            
            # Convert to the format expected by the app
            imported_labels = []
            for _, row in labeled_df.iterrows():
                # Extract attacker and victim labels
                attacker_labels = []
                victim_labels = []
                
                # Handle different possible column names
                if 'attacker_label' in row:
                    if pd.notna(row['attacker_label']) and row['attacker_label']:
                        attacker_labels = [label.strip() for label in str(row['attacker_label']).split(',')]
                
                if 'victim_label' in row:
                    if pd.notna(row['victim_label']) and row['victim_label']:
                        victim_labels = [label.strip() for label in str(row['victim_label']).split(',')]
                
                # Also check for individual label columns
                for col in row.index:
                    if col.startswith('attacker_') and col != 'attacker_label' and col != 'attacker_name':
                        if pd.notna(row[col]) and row[col] == 1:
                            label = col.replace('attacker_', '')
                            if label not in attacker_labels:
                                attacker_labels.append(label)
                    
                    if col.startswith('victim_') and col != 'victim_label' and col != 'victim_name':
                        if pd.notna(row[col]) and row[col] == 1:
                            label = col.replace('victim_', '')
                            if label not in victim_labels:
                                victim_labels.append(label)
                
                # Create labeled kill entry
                labeled_kill = {
                    'attacker_name': row.get('attacker_name', 'Unknown'),
                    'victim_name': row.get('victim_name', 'Unknown'),
                    'kill_tick': row.get('tick', row.get('kill_tick', 0)),
                    'attacker_labels': attacker_labels,
                    'victim_labels': victim_labels,
                    'attacker_label': ', '.join(attacker_labels),
                    'victim_label': ', '.join(victim_labels),
                    # Add other context data if available
                    'distance_xy': row.get('distance_xy', 0),
                    'time_in_round_s': row.get('time_in_round_s', 0),
                    'headshot': row.get('headshot', False),
                    'place': row.get('place', row.get('attacker_place', 'Unknown')),
                    'side': row.get('side', row.get('attacker_side', 'Unknown')),
                }
                
                imported_labels.append(labeled_kill)
            
            # Store in session state
            if imported_labels:
                st.session_state.labeled_data = imported_labels
                st.sidebar.success(f"✅ Imported {len(imported_labels)} labeled kills!")
                st.sidebar.info("Your existing labels are now loaded and ready for active learning!")
                
                # Show label distribution
                st.sidebar.markdown("**Imported Label Distribution:**")
                attacker_counts = {}
                victim_counts = {}
                
                for kill in imported_labels:
                    for label in kill.get('attacker_labels', []):
                        attacker_counts[label] = attacker_counts.get(label, 0) + 1
                    for label in kill.get('victim_labels', []):
                        victim_counts[label] = victim_counts.get(label, 0) + 1
                
                if attacker_counts:
                    st.sidebar.markdown("**Attacker Labels:**")
                    for label, count in attacker_counts.items():
                        st.sidebar.markdown(f"- {label}: {count}")
                
                if victim_counts:
                    st.sidebar.markdown("**Victim Labels:**")
                    for label, count in victim_counts.items():
                        st.sidebar.markdown(f"- {label}: {count}")
            
        except Exception as e:
            st.sidebar.error(f"Error importing labeled data: {str(e)}")
            st.sidebar.info("Make sure your CSV has the expected columns (attacker_name, victim_name, tick, attacker_label, victim_label, etc.)")


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


def create_batch_labeling_controls(batch_kills: pd.DataFrame, kill_contexts: List[Dict], 
                                 map_data: Dict, map_image_path: str, figure_size: Tuple[int, int]) -> Dict[int, Dict]:
    """
    Create batch labeling controls for multiple kills.
    
    Args:
        batch_kills: DataFrame with batch of kills
        kill_contexts: List of kill context dictionaries
        map_data: Map coordinate data
        map_image_path: Path to map image
        figure_size: Figure size for map display
        
    Returns:
        Dictionary mapping kill indices to labels
    """
    batch_labels = {}
    
    st.subheader(f"📦 Batch Labeling ({len(batch_kills)} kills)")
    
    # Create columns for batch display
    cols = st.columns(2)
    
    for i, (_, kill_row) in enumerate(batch_kills.iterrows()):
        col_idx = i % 2
        with cols[col_idx]:
            st.markdown(f"**Kill {i+1}**")
            
            # Display basic kill info
            context = kill_contexts[i]
            st.write(f"**{context['attacker_name']}** → **{context['victim_name']}**")
            st.write(f"Distance: {context['distance_xy']:.0f} units")
            st.write(f"Time: {context['time_in_round_s']:.1f}s")
            
            # Map visualization (smaller)
            if map_data and map_image_path:
                try:
                    attacker_pos = (context['attacker_image_x'], context['attacker_image_y'])
                    victim_pos = (context['victim_image_x'], context['victim_image_y'])
                    
                    # Create smaller map figure
                    fig, ax = plt.subplots(figsize=(4, 3))
                    
                    # Load and display map
                    map_img = Image.open(map_image_path)
                    ax.imshow(map_img)
                    
                    # Plot positions
                    ax.scatter(attacker_pos[0], attacker_pos[1], c='red', s=50, marker='o', label=context['attacker_name'])
                    ax.scatter(victim_pos[0], victim_pos[1], c='blue', s=50, marker='x', label=context['victim_name'])
                    
                    ax.set_title(f"Kill {i+1}")
                    ax.legend(fontsize=6)
                    ax.axis('off')
                    
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.write(f"Map error: {e}")
            
            # Label controls
            attacker_labels = st.multiselect(
                f"Attacker Labels {i+1}",
                ["good_decision", "bad_decision", "precise", "imprecise", "good_positioning", "bad_positioning", "other"],
                key=f"attacker_batch_{i}"
            )
            
            victim_labels = st.multiselect(
                f"Victim Labels {i+1}",
                ["exposed", "no_cover", "good_position", "mistake", "bad_clearing", "other"],
                key=f"victim_batch_{i}"
            )
            
            # Store labels
            if attacker_labels or victim_labels:
                batch_labels[i] = {
                    'attacker': attacker_labels,
                    'victim': victim_labels
                }
            
            st.markdown("---")
    
    return batch_labels


def create_ml_training_controls(filtered_kills: pd.DataFrame, labeled_data: List[Dict]) -> Dict[str, Any]:
    """
    Create ML training controls and suggestions.
    
    Args:
        filtered_kills: DataFrame with filtered kills
        labeled_data: List of labeled kills
        
    Returns:
        Dictionary with ML controls and suggestions
    """
    ml_controls = {}
    
    st.sidebar.markdown("### 🤖 ML Training Features")
    
    # Pre-labeling options
    st.sidebar.markdown("**Pre-labeling Rules:**")
    enable_prelabel = st.sidebar.checkbox("Enable Pre-labeling", value=True)
    
    if enable_prelabel:
        ml_controls['prelabel_rules'] = {
            'headshot_auto_good': st.sidebar.checkbox("Auto-label headshots as 'precise'", value=True),
            'long_distance_auto_good': st.sidebar.checkbox("Auto-label long-distance kills as 'precise'", value=True),
            'close_distance_auto_bad': st.sidebar.checkbox("Auto-label very close kills as 'bad_positioning'", value=True),
            'flash_kills_auto_good': st.sidebar.checkbox("Auto-label flash-assisted kills as 'good_decision'", value=True),
        }
    
    # Active learning options
    st.sidebar.markdown("**Active Learning:**")
    enable_active_learning = st.sidebar.checkbox("Enable Active Learning", value=False)
    
    if enable_active_learning and len(labeled_data) > 10:
        ml_controls['active_learning'] = {
            'uncertainty_threshold': st.sidebar.slider("Uncertainty Threshold", 0.1, 0.9, 0.5),
            'sample_size': st.sidebar.slider("Sample Size", 5, 20, 10),
            'prioritize_uncertain': st.sidebar.checkbox("Prioritize Uncertain Samples", value=True),
        }
        
        # Show active learning suggestions
        st.sidebar.markdown("**Active Learning Suggestions:**")
        
        # Check if we have enough labeled data
        if len(labeled_data) < 10:
            st.sidebar.warning("Need at least 10 labeled kills for active learning")
        else:
            st.sidebar.info(f"Ready to train on {len(labeled_data)} labeled kills")
            
            if st.sidebar.button("🔄 Train Model & Generate Suggestions"):
                try:
                    # Import here to avoid circular imports
                    import lightgbm as lgb
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import LabelEncoder
                    import numpy as np
                    
                    # Prepare training data
                    training_features = []
                    training_labels = []
                    
                    for kill in labeled_data:
                        # Extract features (you can enhance this)
                        features = [
                            kill.get('distance_xy', 0),
                            kill.get('time_in_round_s', 0),
                            kill.get('attacker_health', 100),
                            kill.get('victim_health', 100),
                            1 if kill.get('headshot', False) else 0,
                            1 if kill.get('victim_was_aware', False) else 0,
                            1 if kill.get('had_sound_cue', False) else 0,
                            kill.get('utility_count', 0),
                            kill.get('approach_align_deg', 0) or 0,
                        ]
                        
                        # Use attacker labels as target (simplified - you can enhance this)
                        if 'attacker_labels' in kill and kill['attacker_labels']:
                            # Take the first label for simplicity
                            label = kill['attacker_labels'][0]
                        else:
                            label = 'other'  # Default label
                        
                        training_features.append(features)
                        training_labels.append(label)
                    
                    # Encode labels
                    label_encoder = LabelEncoder()
                    encoded_labels = label_encoder.fit_transform(training_labels)
                    
                    # Split data
                    X_train, X_val, y_train, y_val = train_test_split(
                        training_features, encoded_labels, test_size=0.2, random_state=42
                    )
                    
                    # Train LightGBM model
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42,
                        verbose=-1
                    )
                    
                    model.fit(X_train, y_train)
                    
                    # Predict on unlabeled data
                    unlabeled_features = []
                    unlabeled_indices = []
                    
                    for idx, (_, kill_row) in enumerate(filtered_kills.iterrows()):
                        # Check if this kill is already labeled
                        is_labeled = any(
                            labeled_kill.get('tick') == kill_row.get('tick') and 
                            labeled_kill.get('attacker_name') == kill_row.get('attacker_name')
                            for labeled_kill in labeled_data
                        )
                        
                        if not is_labeled:
                            # Extract features for unlabeled kill
                            features = [
                                kill_row.get('distance_xy', 0),
                                kill_row.get('time_in_round_s', 0),
                                kill_row.get('attacker_health', 100),
                                kill_row.get('victim_health', 100),
                                1 if kill_row.get('headshot', False) else 0,
                                1 if kill_row.get('victim_was_aware', False) else 0,
                                1 if kill_row.get('had_sound_cue', False) else 0,
                                kill_row.get('utility_count', 0),
                                kill_row.get('approach_align_deg', 0) or 0,
                            ]
                            
                            unlabeled_features.append(features)
                            unlabeled_indices.append(idx)
                    
                    if unlabeled_features:
                        # Get prediction probabilities
                        probabilities = model.predict_proba(unlabeled_features)
                        
                        # Calculate uncertainty (entropy)
                        uncertainties = []
                        for prob in probabilities:
                            # Calculate entropy: -sum(p * log(p))
                            entropy = -np.sum(prob * np.log(prob + 1e-10))
                            uncertainties.append(entropy)
                        
                        # Sort by uncertainty (highest first)
                        uncertainty_data = list(zip(unlabeled_indices, uncertainties))
                        uncertainty_data.sort(key=lambda x: x[1], reverse=True)
                        
                        # Calculate accuracy on validation set
                        val_accuracy = model.score(X_val, y_val)
                        
                        # Store suggestions in session state
                        st.session_state['active_learning_suggestions'] = {
                            'suggested_indices': [idx for idx, _ in uncertainty_data[:ml_controls['active_learning']['sample_size']]],
                            'uncertainties': [unc for _, unc in uncertainty_data[:ml_controls['active_learning']['sample_size']]],
                            'model_accuracy': val_accuracy
                        }
                        
                        st.sidebar.success(f"✅ Model trained! Accuracy: {val_accuracy:.2f}")
                        st.sidebar.info(f"📊 Found {len(unlabeled_features)} unlabeled kills")
                        st.sidebar.info(f"🎯 Top {ml_controls['active_learning']['sample_size']} uncertain samples ready")
                        
                        # Show top suggestions
                        st.sidebar.markdown("**Top Uncertain Kills:**")
                        for i, (idx, uncertainty) in enumerate(uncertainty_data[:5]):
                            kill_row = filtered_kills.iloc[idx]
                            st.sidebar.markdown(f"{i+1}. **{kill_row.get('attacker_name', 'Unknown')}** → **{kill_row.get('victim_name', 'Unknown')}** (uncertainty: {uncertainty:.3f})")
                    
                    else:
                        st.sidebar.warning("No unlabeled kills found!")
                        
                except Exception as e:
                    st.sidebar.error(f"Error training model: {str(e)}")
                    st.sidebar.info("Make sure you have enough labeled data with consistent features")
    
    # Quick labeling templates
    st.sidebar.markdown("**Quick Labeling Templates:**")
    template = st.sidebar.selectbox(
        "Template",
        ["", "AWP Long Distance", "Rush Entry", "Trade Kill", "Clutch Situation", "Eco Round"]
    )
    
    if template:
        ml_controls['template'] = template
        st.sidebar.markdown(f"**Template: {template}**")
        
        if template == "AWP Long Distance":
            st.sidebar.markdown("- Attacker: precise, good_decision")
            st.sidebar.markdown("- Victim: exposed, mistake")
        elif template == "Rush Entry":
            st.sidebar.markdown("- Attacker: bad_decision, imprecise")
            st.sidebar.markdown("- Victim: good_position, no_cover")
        elif template == "Trade Kill":
            st.sidebar.markdown("- Attacker: good_decision, precise")
            st.sidebar.markdown("- Victim: exposed, mistake")
        elif template == "Clutch Situation":
            st.sidebar.markdown("- Attacker: good_decision, precise")
            st.sidebar.markdown("- Victim: exposed, mistake")
        elif template == "Eco Round":
            st.sidebar.markdown("- Attacker: bad_decision, imprecise")
            st.sidebar.markdown("- Victim: good_position, no_cover")
    
    # Keyboard shortcuts info
    st.sidebar.markdown("**Keyboard Shortcuts:**")
    st.sidebar.markdown("- `1-7`: Quick attacker labels")
    st.sidebar.markdown("- `Q-T`: Quick victim labels")
    st.sidebar.markdown("- `Space`: Save and next")
    st.sidebar.markdown("- `Backspace`: Previous kill")
    
    # Progress tracking
    if labeled_data:
        progress = len(labeled_data) / len(filtered_kills) * 100
        st.sidebar.markdown(f"**Progress: {progress:.1f}%**")
        st.sidebar.progress(progress / 100)
        
        # Label distribution
        st.sidebar.markdown("**Label Distribution:**")
        attacker_labels = []
        for d in labeled_data:
            if 'attacker_labels' in d and d['attacker_labels']:
                attacker_labels.extend(d['attacker_labels'])
        
        victim_labels = []
        for d in labeled_data:
            if 'victim_labels' in d and d['victim_labels']:
                victim_labels.extend(d['victim_labels'])
        
        if attacker_labels:
            attacker_counts = pd.Series(attacker_labels).value_counts()
            for label, count in attacker_counts.items():
                st.sidebar.markdown(f"- {label}: {count}")
        
        if victim_labels:
            victim_counts = pd.Series(victim_labels).value_counts()
            for label, count in victim_counts.items():
                st.sidebar.markdown(f"- {label}: {count}")
    
    return ml_controls


def display_enhanced_kill_info(kill_context: Dict) -> None:
    """
    Display enhanced kill information with additional context.
    
    Args:
        kill_context: Enhanced kill context dictionary
    """
    st.subheader("🎯 Enhanced Kill Analysis")
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information:**")
        st.write(f"**Attacker:** {kill_context['attacker_name']}")
        st.write(f"**Victim:** {kill_context['victim_name']}")
        st.write(f"**Distance:** {kill_context['distance_xy']:.0f} units")
        st.write(f"**Headshot:** {'Yes' if kill_context['headshot'] else 'No'}")
        st.write(f"**Time in Round:** {kill_context['time_in_round_s']:.1f}s")
        
        # Round context
        if 'round_number' in kill_context and kill_context['round_number'] is not None:
            st.write(f"**Round:** {kill_context['round_number']}")
        if 'round_phase' in kill_context:
            st.write(f"**Phase:** {kill_context['round_phase']}")
        if 'bomb_planted' in kill_context and kill_context['bomb_planted']:
            st.write(f"**Bomb Planted:** Yes ({kill_context.get('time_since_bomb_plant', 0):.1f}s ago)")
    
    with col2:
        st.markdown("**Player States:**")
        st.write(f"**Attacker Health:** {kill_context.get('attacker_health', 100)}")
        st.write(f"**Victim Health:** {kill_context.get('victim_health', 100)}")
        st.write(f"**Attacker Moving:** {'Yes' if kill_context.get('attacker_is_moving', False) else 'No'}")
        st.write(f"**Victim Moving:** {'Yes' if kill_context.get('victim_is_moving', False) else 'No'}")
        st.write(f"**Attacker Ducking:** {'Yes' if kill_context.get('attacker_is_ducking', False) else 'No'}")
        st.write(f"**Victim Ducking:** {'Yes' if kill_context.get('victim_is_ducking', False) else 'No'}")
    
    # Sound cues
    if 'had_sound_cue' in kill_context:
        st.markdown("**🔊 Sound Analysis:**")
        sound_col1, sound_col2 = st.columns(2)
        
        with sound_col1:
            st.write(f"**Had Sound Cue:** {'Yes' if kill_context['had_sound_cue'] else 'No'}")
            if kill_context['had_sound_cue']:
                st.write(f"**Sound Types:** {', '.join(kill_context.get('sound_cue_types', []))}")
                st.write(f"**Sound Count:** {kill_context.get('sound_cue_count', 0)}")
        
        with sound_col2:
            if kill_context.get('time_since_last_sound') is not None:
                st.write(f"**Time Since Sound:** {kill_context['time_since_last_sound']:.1f}s")
            if kill_context.get('attacker_distance_when_heard') is not None:
                st.write(f"**Distance When Heard:** {kill_context['attacker_distance_when_heard']:.0f} units")
            st.write(f"**Attacker Visible:** {'Yes' if kill_context.get('attacker_visible', False) else 'No'}")
    
    # Victim awareness analysis
    if 'victim_was_aware' in kill_context:
        st.markdown("**👁️ Victim Awareness Analysis:**")
        awareness_col1, awareness_col2 = st.columns(2)
        
        with awareness_col1:
            st.write(f"**Victim Was Aware:** {'Yes' if kill_context['victim_was_aware'] else 'No'}")
            st.write(f"**Victim Was Watching:** {'Yes' if kill_context.get('victim_was_watching', False) else 'No'}")
            st.write(f"**Victim Was Backstabbed:** {'Yes' if kill_context.get('victim_was_backstabbed', False) else 'No'}")
            confidence = kill_context.get('awareness_confidence', 0)
            st.write(f"**Awareness Confidence:** {confidence:.1%}")
        
        with awareness_col2:
            if kill_context.get('angle_to_attacker') is not None:
                st.write(f"**Angle to Attacker:** {kill_context['angle_to_attacker']:.1f}°")
            if kill_context.get('victim_view_angle') is not None:
                st.write(f"**Victim View Angle:** {kill_context['victim_view_angle']:.1f}°")
            if kill_context.get('angle_difference') is not None:
                st.write(f"**Angle Difference:** {kill_context['angle_difference']:.1f}°")
            
            # Add interpretation
            if kill_context.get('angle_to_attacker') is not None:
                angle = kill_context['angle_to_attacker']
                if angle > 135 and angle < 225:
                    st.write("**Position:** Behind victim (backstab)")
                elif angle > 45 and angle < 135:
                    st.write("**Position:** To victim's right")
                elif angle > 225 and angle < 315:
                    st.write("**Position:** To victim's left")
                else:
                    st.write("**Position:** In front of victim")
            
            # Show awareness timing info
            if kill_context.get('time_since_last_sight') is not None:
                st.write(f"**Time Since Last Sight:** {kill_context['time_since_last_sight']:.1f}s")
            if kill_context.get('awareness_detected_at_tick') is not None:
                st.write(f"**Awareness Detected At:** Tick {kill_context['awareness_detected_at_tick']}")
    
    # Utility context
    if 'utility_count' in kill_context and kill_context['utility_count'] > 0:
        st.markdown("**💣 Utility Context:**")
        utility_col1, utility_col2 = st.columns(2)
        
        with utility_col1:
            st.write(f"**Active Utility:** {kill_context['utility_count']}")
            st.write(f"**Flash Active:** {'Yes' if kill_context.get('flash_active', False) else 'No'}")
            st.write(f"**Smoke Active:** {'Yes' if kill_context.get('smoke_active', False) else 'No'}")
            st.write(f"**Molotov Active:** {'Yes' if kill_context.get('molotov_active', False) else 'No'}")
        
        with utility_col2:
            if kill_context.get('closest_utility_distance') is not None:
                st.write(f"**Closest Utility:** {kill_context['closest_utility_distance']:.0f} units")
            if kill_context.get('utility_thrower'):
                st.write(f"**Utility Thrower:** {kill_context['utility_thrower']}")
            if kill_context.get('time_since_utility') is not None:
                st.write(f"**Time Since Utility:** {kill_context['time_since_utility']:.1f}s")
            st.write(f"**Utility Affecting Kill:** {'Yes' if kill_context.get('utility_affecting_kill', False) else 'No'}")
    
    # Tactical analysis
    st.markdown("**🎯 Tactical Analysis:**")
    tactical_col1, tactical_col2, tactical_col3 = st.columns(3)
    
    with tactical_col1:
        st.write(f"**Kill Advantage:** {kill_context.get('kill_advantage', 0)}")
        st.write(f"**Distance Category:** {kill_context.get('distance_category', 'unknown')}")
        approach_align = kill_context.get('approach_align_deg')
        if approach_align is not None:
            st.write(f"**Approach Alignment:** {approach_align:.1f}°")
        else:
            st.write("**Approach Alignment:** Not moving")
    
    with tactical_col2:
        st.write(f"**Eco Kill:** {'Yes' if kill_context.get('is_eco_kill', False) else 'No'}")
        st.write(f"**Trade Kill:** {'Yes' if kill_context.get('is_trade_kill', False) else 'No'}")
        st.write(f"**Clutch Situation:** {'Yes' if kill_context.get('is_clutch_situation', False) else 'No'}")
    
    with tactical_col3:
        st.write(f"**Attacker Has Primary:** {'Yes' if kill_context.get('attacker_has_primary', False) else 'No'}")
        st.write(f"**Victim Has Primary:** {'Yes' if kill_context.get('victim_has_primary', False) else 'No'}")
        st.write(f"**Attacker Has Utility:** {'Yes' if kill_context.get('attacker_has_utility', False) else 'No'}")
