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
            [
                # Essential Performance Labels
                "precise", "good_positioning", "bad_positioning",
                
                # Essential Decision Making Labels
                "good_utility_usage", "bad_utility_usage", "good_peek", "bad_peek",
                
                # Essential Situational Labels
                "clutch_play", "choke", "entry_frag",
                
                # Legacy Labels (keeping for compatibility)
                "good_decision", "bad_decision", "imprecise", "other"
            ],
            key="attacker_labels"
        )
    
    with col2:
        st.write("**Victim Labels** (Multiple selection)")
        victim_labels = st.multiselect(
            "Victim",
            [
                # Essential Victim Labels
                "bad_clearing", "exposed", "no_cover", "good_position", "mistake", "other"
            ],
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
    
    # Match and Round Information
    if context.get('round_number') is not None:
        st.markdown("**🎮 Match & Round Info**")
        col_match, col_round = st.columns(2)
        
        with col_match:
            team1 = context.get('team1_name', 'Team 1')
            team2 = context.get('team2_name', 'Team 2')
            score_t = context.get('match_score_t', 0)
            score_ct = context.get('match_score_ct', 0)
            st.write(f"**Match Score:** {team1} {score_t} - {score_ct} {team2}")
        
        with col_round:
            round_num = context.get('round_number', 0)
            round_phase = context.get('round_phase', 'unknown')
            st.write(f"**Round:** {round_num} ({round_phase})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Info**")
        st.write(f"Attacker: {context.get('attacker_name', 'Unknown')}")
        st.write(f"Victim: {context.get('victim_name', 'Unknown')}")
        st.write(f"Attacker Side: {context.get('side', 'Unknown')}")
        st.write(f"Attacker Place: {context.get('place', 'Unknown')}")
        st.write(f"Headshot: {context.get('headshot', False)}")
        
        # Weapon information
        st.write("**Weapons**")
        attacker_weapon = context.get('attacker_weapon', 'Unknown')
        victim_weapon = context.get('victim_weapon', 'Unknown')
        st.write(f"Attacker Weapon: {attacker_weapon}")
        st.write(f"Victim Weapon: {victim_weapon}")
        
        # Equipment advantage analysis
        if attacker_weapon != 'Unknown' and victim_weapon != 'Unknown':
            st.write("**Equipment Analysis:**")
            # Simple weapon tier analysis
            primary_weapons = ['ak47', 'm4a1', 'awp', 'sg553', 'aug', 'famas', 'galil']
            secondary_weapons = ['deagle', 'usp', 'glock', 'p250', 'tec9', 'cz75']
            
            attacker_has_primary = any(weapon in attacker_weapon.lower() for weapon in primary_weapons)
            victim_has_primary = any(weapon in victim_weapon.lower() for weapon in primary_weapons)
            
            if attacker_has_primary and not victim_has_primary:
                st.write("✅ Attacker has equipment advantage")
            elif not attacker_has_primary and victim_has_primary:
                st.write("❌ Attacker has equipment disadvantage")
            else:
                st.write("⚖️ Equipment is balanced")
    
    with col2:
        st.write("**Context**")
        st.write(f"Time in Round: {context.get('time_in_round_s', 0):.1f}s")
        st.write(f"Distance: {context.get('distance_xy', 0):.0f} units")
        approach_align = context.get('approach_align_deg')
        if approach_align is not None:
            st.write(f"Approach Alignment: {approach_align:.1f}°")
            # Add interpretation
            if approach_align < 30:
                st.write("🎯 Excellent alignment (moving directly toward victim)")
            elif approach_align < 60:
                st.write("✅ Good alignment (moving toward victim)")
            elif approach_align < 90:
                st.write("⚠️ Fair alignment (somewhat toward victim)")
            elif approach_align < 120:
                st.write("❌ Poor alignment (moving sideways)")
            else:
                st.write("❌ Very poor alignment (moving away from victim)")
        else:
            st.write("Approach Alignment: Not moving (stationary)")
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


def create_file_uploaders() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], 
                                    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], 
                                    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Create file upload widgets with bulk upload support.
    
    Returns:
        Tuple of (kills_df, ticks_df, grenades_df, damages_df, shots_df, smokes_df, infernos_df, bomb_df)
    """
    st.header("Upload Data Files")
    
    # Bulk upload option
    upload_method = st.radio(
        "Upload Method:",
        ["Bulk Upload (All files at once)", "Individual Upload"],
        key="upload_method"
    )
    
    kills_df = None
    ticks_df = None
    grenades_df = None
    damages_df = None
    shots_df = None
    smokes_df = None
    infernos_df = None
    bomb_df = None
    rounds_df = None
    
    if upload_method == "Bulk Upload (All files at once)":
        st.markdown("**📁 Bulk Upload - Select all your .parquet files:**")
        st.info("💡 **Tip:** You can select multiple files by holding Ctrl (or Cmd on Mac) while clicking, or drag and drop all your .parquet files at once!")
        
        # Show expected file types
        with st.expander("📋 Expected File Types"):
            st.markdown("""
            **Required Files:**
            - `kills.parquet` - Kill events data
            - `ticks.parquet` - Player position and state data
            
            **Optional Files (for enhanced analysis):**
            - `grenades.parquet` - Grenade events
            - `damages.parquet` - Damage events
            - `shots.parquet` - Shot events
            - `smokes.parquet` - Smoke events
            - `infernos.parquet` - Molotov/incendiary events
            - `bomb.parquet` - Bomb events
            - `rounds.parquet` - Round information
            """)
        
        uploaded_files = st.file_uploader(
            "Upload all .parquet files at once",
            type=['parquet'],
            accept_multiple_files=True,
            key="bulk_upload"
        )
        
        if uploaded_files:
            st.success(f"📦 Uploaded {len(uploaded_files)} files")
            
            # Process each uploaded file
            for uploaded_file in uploaded_files:
                try:
                    filename = uploaded_file.name.lower()
                    
                    if 'kill' in filename:
                        kills_df = pd.read_parquet(uploaded_file)
                        st.success(f"✅ Loaded kills: {len(kills_df)} kills from {uploaded_file.name}")
                    elif 'tick' in filename:
                        ticks_df = pd.read_parquet(uploaded_file)
                        st.success(f"✅ Loaded ticks: {len(ticks_df)} ticks from {uploaded_file.name}")
                    elif 'grenade' in filename or 'flash' in filename or 'smoke' in filename or 'molotov' in filename:
                        grenades_df = pd.read_parquet(uploaded_file)
                        st.success(f"✅ Loaded grenades: {len(grenades_df)} events from {uploaded_file.name}")
                    elif 'round' in filename:
                        rounds_df = pd.read_parquet(uploaded_file)
                        st.success(f"📋 Loaded rounds: {len(rounds_df)} rounds from {uploaded_file.name}")
                    elif 'bomb' in filename:
                        bomb_df = pd.read_parquet(uploaded_file)
                        st.success(f"💣 Loaded bomb: {len(bomb_df)} events from {uploaded_file.name}")
                    elif 'damage' in filename:
                        damages_df = pd.read_parquet(uploaded_file)
                        st.success(f"💥 Loaded damage: {len(damages_df)} events from {uploaded_file.name}")
                    elif 'shot' in filename:
                        shots_df = pd.read_parquet(uploaded_file)
                        st.success(f"🔫 Loaded shots: {len(shots_df)} events from {uploaded_file.name}")
                    elif 'smoke' in filename:
                        smokes_df = pd.read_parquet(uploaded_file)
                        st.success(f"💨 Loaded smokes: {len(smokes_df)} events from {uploaded_file.name}")
                    elif 'inferno' in filename or 'molotov' in filename:
                        infernos_df = pd.read_parquet(uploaded_file)
                        st.success(f"🔥 Loaded infernos: {len(infernos_df)} events from {uploaded_file.name}")
                    else:
                        st.warning(f"❓ Unknown file type: {uploaded_file.name}")
                        
                except Exception as e:
                    st.error(f"❌ Error loading {uploaded_file.name}: {e}")
            
            # Show summary
            if kills_df is not None or ticks_df is not None or grenades_df is not None:
                st.markdown("**📊 Upload Summary:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if kills_df is not None:
                        st.success(f"Kills: {len(kills_df)}")
                    else:
                        st.error("Kills: Missing")
                    
                    if ticks_df is not None:
                        st.success(f"Ticks: {len(ticks_df)}")
                    else:
                        st.error("Ticks: Missing")
                
                with col2:
                    if grenades_df is not None:
                        st.success(f"Grenades: {len(grenades_df)}")
                    else:
                        st.info("Grenades: Optional")
                    
                    if damages_df is not None:
                        st.success(f"Damage: {len(damages_df)}")
                    else:
                        st.info("Damage: Optional")
                
                with col3:
                    if shots_df is not None:
                        st.success(f"Shots: {len(shots_df)}")
                    else:
                        st.info("Shots: Optional")
                    
                    if bomb_df is not None:
                        st.success(f"Bomb: {len(bomb_df)}")
                    else:
                        st.info("Bomb: Optional")
                
                with col4:
                    if smokes_df is not None:
                        st.success(f"Smokes: {len(smokes_df)}")
                    else:
                        st.info("Smokes: Optional")
                    
                    if infernos_df is not None:
                        st.success(f"Infernos: {len(infernos_df)}")
                    else:
                        st.info("Infernos: Optional")
    
    else:
        # Individual upload (original method)
        st.markdown("**📁 Individual Upload:**")
        
        # Kills file (required)
        kills_file = st.file_uploader(
            "Upload kills.parquet (required)", 
            type=['parquet'],
            key="kills_upload"
        )
        
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
        
        if grenades_file is not None:
            try:
                grenades_df = pd.read_parquet(grenades_file)
                st.success(f"Loaded {len(grenades_df)} grenade events")
            except Exception as e:
                st.error(f"Error loading grenades file: {e}")
    
    return kills_df, ticks_df, grenades_df, damages_df, shots_df, smokes_df, infernos_df, bomb_df, rounds_df


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
    st.sidebar.markdown("**🎯 Active Learning:**")
    enable_active_learning = st.sidebar.checkbox("Enable Active Learning", value=False)
    
    if enable_active_learning:
        st.sidebar.markdown("🟢 **Active Learning Enabled**")
        st.sidebar.markdown("• Model will suggest uncertain kills to label")
        st.sidebar.markdown("• Auto-retrains when you add new labels")
        st.sidebar.markdown("• Prioritizes kills that improve the model most")
    else:
        st.sidebar.markdown("🔴 **Active Learning Disabled**")
        st.sidebar.markdown("• Manual labeling only")
        st.sidebar.markdown("• No automatic suggestions")
    
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
            st.sidebar.warning("⚠️ **Need at least 10 labeled kills for active learning**")
            st.sidebar.markdown(f"• Current: {len(labeled_data)} labeled kills")
            st.sidebar.markdown(f"• Required: 10+ labeled kills")
        else:
            st.sidebar.success(f"✅ **Ready to Train!**")
            st.sidebar.markdown(f"• {len(labeled_data)} labeled kills available")
            st.sidebar.markdown(f"• Click 'Train Model' to start active learning")
            
            # Show current model status
            if 'active_learning_suggestions' in st.session_state:
                suggestions = st.session_state['active_learning_suggestions']
                if 'model_accuracy' in suggestions:
                    accuracy = suggestions['model_accuracy']
                    labeled_count = suggestions.get('labeled_count', 0)
                    current_labeled = len(labeled_data)
                    total_unlabeled = suggestions.get('total_unlabeled', 0)
                    error = suggestions.get('error', None)
                    
                    # Show model status with color coding
                    if error:
                        st.sidebar.error(f"❌ **Model Error:** {error}")
                    elif current_labeled > labeled_count:
                        st.sidebar.warning(f"🔄 **Model Outdated!**")
                        st.sidebar.markdown(f"• {current_labeled - labeled_count} new labels added")
                        st.sidebar.markdown(f"• Accuracy: {accuracy:.1%} (outdated)")
                        st.sidebar.markdown(f"• **Auto-retrain will happen when you save labels**")
                        st.sidebar.markdown(f"• Or click 'Train Model' to update now")
                    else:
                        st.sidebar.success(f"🤖 **Model Active**")
                        st.sidebar.markdown(f"• Accuracy: {accuracy:.1%}")
                        st.sidebar.markdown(f"• Trained on {labeled_count} labels")
                        if total_unlabeled > 0:
                            st.sidebar.markdown(f"• {total_unlabeled} unlabeled kills available")
                        st.sidebar.markdown(f"• Ready to suggest uncertain kills")
                        
                        # Show how uncertainty works
                        st.sidebar.markdown("**💡 How Uncertainty Works:**")
                        st.sidebar.markdown("• **High uncertainty** = Model is confused about this kill")
                        st.sidebar.markdown("• **Low uncertainty** = Model is confident in its prediction")
                        st.sidebar.markdown("• **Label uncertain kills** to improve the model most")
                        st.sidebar.markdown("• **After labeling, uncertainty should change**")
                        
                        # Add explanation about why uncertainty might not change
                        st.sidebar.markdown("**⚠️ Why Uncertainty Might Not Change:**")
                        st.sidebar.markdown("• **Limited features:** Your CSV only has basic features (distance, time, headshot)")
                        st.sidebar.markdown("• **Similar kills:** Many kills might have similar characteristics")
                        st.sidebar.markdown("• **Model needs more data:** 119 kills might not be enough variety")
                        st.sidebar.markdown("• **Feature quality:** Missing enhanced features limits learning")
                        
                        st.sidebar.markdown("**💡 To Improve Results:**")
                        st.sidebar.markdown("• Label more diverse kills (different distances, times, situations)")
                        st.sidebar.markdown("• Use enhanced data files (with awareness, utility, etc.)")
                        st.sidebar.markdown("• Try different label combinations")
                        st.sidebar.markdown("• The model will improve as you add more labeled data")
                else:
                    st.sidebar.info("🤖 Model not yet trained")
            else:
                st.sidebar.info("🤖 Model not yet trained")
            
            if st.sidebar.button("🔄 Train Model & Generate Suggestions"):
                try:
                    # Show training status with more detailed progress
                    with st.sidebar.status("🤖 Training ML Model...", expanded=True) as status:
                        st.sidebar.write("📊 **Step 1:** Preparing training data...")
                        
                        # Import here to avoid circular imports
                        try:
                            import lightgbm as lgb
                            use_lightgbm = True
                            st.sidebar.write("✅ Using LightGBM model (faster & more accurate)")
                        except ImportError:
                            from sklearn.ensemble import RandomForestClassifier
                            use_lightgbm = False
                            st.sidebar.write("⚠️ LightGBM not available, using Random Forest instead")
                        
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import LabelEncoder
                        import numpy as np
                        
                        st.sidebar.write("🔍 **Step 2:** Extracting features from labeled data...")
                        
                        # Prepare training data
                        training_features = []
                        training_labels = []
                    
                        # SIMPLIFIED FEATURE EXTRACTION - Works with imported CSV data
                        st.sidebar.write("🔍 **Step 2:** Extracting features from labeled data...")
                        
                        # Check what features are available in the data
                        sample_kill = labeled_data[0] if labeled_data else {}
                        available_features = []
                        
                        # Basic features that should be available
                        basic_features = ['distance_xy', 'time_in_round_s', 'headshot']
                        for feature in basic_features:
                            if feature in sample_kill and sample_kill[feature] is not None:
                                available_features.append(feature)
                        
                        # Enhanced features (optional)
                        enhanced_features = ['victim_was_aware', 'had_sound_cue', 'utility_count', 'approach_align_deg']
                        for feature in enhanced_features:
                            if feature in sample_kill and sample_kill[feature] is not None:
                                available_features.append(feature)
                        
                        st.sidebar.write(f"📋 **Available Features:** {', '.join(available_features)}")
                        
                        # Extract features based on what's available
                        training_features = []
                        training_labels = []
                        
                        for kill in labeled_data:
                            features = []
                            
                            # Always include basic features with fallbacks
                            features.append(float(kill.get('distance_xy', 0)))
                            features.append(float(kill.get('time_in_round_s', 0)))
                            features.append(1 if kill.get('headshot', False) else 0)
                            
                            # Add enhanced features if available, otherwise use defaults
                            if 'victim_was_aware' in available_features:
                                features.append(1 if kill.get('victim_was_aware', False) else 0)
                            else:
                                features.append(0)  # Default: not aware
                            
                            if 'had_sound_cue' in available_features:
                                features.append(1 if kill.get('had_sound_cue', False) else 0)
                            else:
                                features.append(0)  # Default: no sound cue
                            
                            if 'utility_count' in available_features:
                                features.append(float(kill.get('utility_count', 0)))
                            else:
                                features.append(0)  # Default: no utility
                            
                            if 'approach_align_deg' in available_features:
                                features.append(float(kill.get('approach_align_deg', 0) or 0))
                            else:
                                features.append(0)  # Default: no movement
                            
                            # Use attacker labels as target
                            if 'attacker_labels' in kill and kill['attacker_labels']:
                                label = kill['attacker_labels'][0]
                            else:
                                label = 'other'
                            
                            training_features.append(features)
                            training_labels.append(label)
                        
                        st.sidebar.write(f"📈 **Step 3:** Training on {len(training_features)} samples...")
                        st.sidebar.write(f"📋 **Features Used:** {', '.join(available_features)}")
                        
                        # Show label distribution
                        label_counts = {}
                        for label in training_labels:
                            label_counts[label] = label_counts.get(label, 0) + 1
                        st.sidebar.write("📊 **Label Distribution:**")
                        for label, count in label_counts.items():
                            st.sidebar.write(f"   - {label}: {count} samples")
                        
                        # Show feature availability
                        st.sidebar.write("🔍 **Feature Availability:**")
                        for feature in basic_features + enhanced_features:
                            if feature in available_features:
                                st.sidebar.write(f"   ✅ {feature}: Available")
                            else:
                                st.sidebar.write(f"   ❌ {feature}: Using default value")
                    
                    # Encode labels
                    label_encoder = LabelEncoder()
                    encoded_labels = label_encoder.fit_transform(training_labels)
                    
                    # Split data
                    X_train, X_val, y_train, y_val = train_test_split(
                        training_features, encoded_labels, test_size=0.2, random_state=42
                    )
                    
                    st.sidebar.write(f"🔄 **Step 4:** Training model ({len(X_train)} train, {len(X_val)} validation)...")
                    
                    # Train model (LightGBM or Random Forest)
                    if use_lightgbm:
                        model = lgb.LGBMClassifier(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=6,
                            random_state=42,
                            verbose=-1
                        )
                    else:
                        model = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=6,
                            random_state=42
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
                            # Extract features for unlabeled kill (same logic as training)
                            features = []
                            
                            # Always include basic features with fallbacks
                            features.append(float(kill_row.get('distance_xy', 0)))
                            features.append(float(kill_row.get('time_in_round_s', 0)))
                            features.append(1 if kill_row.get('headshot', False) else 0)
                            
                            # Add enhanced features if available, otherwise use defaults
                            if 'victim_was_aware' in available_features:
                                features.append(1 if kill_row.get('victim_was_aware', False) else 0)
                            else:
                                features.append(0)  # Default: not aware
                            
                            if 'had_sound_cue' in available_features:
                                features.append(1 if kill_row.get('had_sound_cue', False) else 0)
                            else:
                                features.append(0)  # Default: no sound cue
                            
                            if 'utility_count' in available_features:
                                features.append(float(kill_row.get('utility_count', 0)))
                            else:
                                features.append(0)  # Default: no utility
                            
                            if 'approach_align_deg' in available_features:
                                features.append(float(kill_row.get('approach_align_deg', 0) or 0))
                            else:
                                features.append(0)  # Default: no movement
                            
                            unlabeled_features.append(features)
                            unlabeled_indices.append(idx)
                    
                    if unlabeled_features:
                        try:
                            # Get prediction probabilities
                            probabilities = model.predict_proba(unlabeled_features)
                            
                            # Calculate uncertainty (entropy)
                            uncertainties = []
                            for prob in probabilities:
                                # Add small epsilon to avoid log(0)
                                prob_safe = prob + 1e-10
                                prob_safe = prob_safe / prob_safe.sum()  # Renormalize
                                # Use log2 for entropy calculation (more standard)
                                entropy = -np.sum(prob_safe * np.log2(prob_safe + 1e-10))
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
                                'model_accuracy': val_accuracy,
                                'labeled_count': len(labeled_data),
                                'total_unlabeled': len(unlabeled_features)
                            }
                            
                            st.sidebar.write(f"✅ **Successfully calculated uncertainties for {len(unlabeled_features)} unlabeled kills**")
                            st.sidebar.write(f"🎯 **Top 3 Uncertain Kills:**")
                            for i, (idx, uncertainty) in enumerate(uncertainty_data[:3]):
                                kill_row = filtered_kills.iloc[idx]
                                st.sidebar.write(f"   {i+1}. {kill_row.get('attacker_name', 'Unknown')} → {kill_row.get('victim_name', 'Unknown')}")
                                st.sidebar.write(f"      Uncertainty: {uncertainty:.3f}")
                            
                            st.sidebar.write(f"📊 **Uncertainty Range:** {min(uncertainties):.3f} - {max(uncertainties):.3f}")
                        except Exception as e:
                            st.sidebar.error(f"❌ **Error calculating uncertainties:** {str(e)}")
                            st.sidebar.write("🔍 **Debug Info:**")
                            st.sidebar.write(f"   - Unlabeled features shape: {len(unlabeled_features)} samples")
                            if unlabeled_features:
                                st.sidebar.write(f"   - Feature vector length: {len(unlabeled_features[0])}")
                                st.sidebar.write(f"   - Sample features: {unlabeled_features[0]}")
                            
                            # Still store model info
                            val_accuracy = model.score(X_val, y_val)
                            st.session_state['active_learning_suggestions'] = {
                                'model_accuracy': val_accuracy,
                                'labeled_count': len(labeled_data),
                                'error': str(e)
                            }
                        
                        st.sidebar.write("🎯 **Step 5:** Calculating uncertainty scores...")
                        
                        # Show top suggestions
                        st.sidebar.markdown("**🎯 Top Uncertain Kills (Prioritize These):**")
                        for i, (idx, uncertainty) in enumerate(uncertainty_data[:5]):
                            kill_row = filtered_kills.iloc[idx]
                            st.sidebar.markdown(f"{i+1}. **{kill_row.get('attacker_name', 'Unknown')}** → **{kill_row.get('victim_name', 'Unknown')}**")
                            st.sidebar.markdown(f"   Uncertainty: {uncertainty:.3f} (Higher = More Uncertain)")
                        
                        status.update(label=f"✅ Model Trained Successfully!", state="complete")
                        st.sidebar.success(f"🎉 **Training Complete!**")
                        st.sidebar.success(f"📊 **Model Accuracy:** {val_accuracy:.1%}")
                        st.sidebar.info(f"📈 **Training Data:** {len(training_features)} labeled kills")
                        st.sidebar.info(f"🎯 **Unlabeled Kills:** {len(unlabeled_features)} available")
                        st.sidebar.info(f"🔍 **Top Uncertain:** {ml_controls['active_learning']['sample_size']} kills prioritized")
                        
                        # Show what the model learned
                        st.sidebar.markdown("**🧠 What the Model Learned:**")
                        st.sidebar.markdown(f"• Trained on {len(label_counts)} different label types")
                        st.sidebar.markdown(f"• Uses {len(training_features[0])} features per kill")
                        st.sidebar.markdown(f"• Validation accuracy: {val_accuracy:.1%}")
                        if use_lightgbm:
                            st.sidebar.markdown("• Using LightGBM (gradient boosting)")
                        else:
                            st.sidebar.markdown("• Using Random Forest (ensemble)")
                    
                    else:
                        status.update(label="⚠️ No unlabeled kills found!", state="error")
                        st.sidebar.warning("No unlabeled kills found!")
                        
                except Exception as e:
                    status.update(label=f"❌ Training failed: {str(e)}", state="error")
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


def auto_retrain_model(labeled_data: List[Dict], filtered_kills: pd.DataFrame = None) -> None:
    """
    Automatically retrain the model with current labeled data.
    
    Args:
        labeled_data: List of labeled kills
        filtered_kills: DataFrame with filtered kills (optional, for uncertainty calculation)
    """
    if len(labeled_data) < 10:
        return  # Not enough data
    
    try:
        # Import here to avoid circular imports
        try:
            import lightgbm as lgb
            use_lightgbm = True
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            use_lightgbm = False
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        # Prepare training data
        training_features = []
        training_labels = []
        
        # Check what features are available in the data
        sample_kill = labeled_data[0] if labeled_data else {}
        available_features = []
        
        # Basic features that should be available
        basic_features = ['distance_xy', 'time_in_round_s', 'headshot']
        for feature in basic_features:
            if feature in sample_kill and sample_kill[feature] is not None:
                available_features.append(feature)
        
        # Enhanced features (optional)
        enhanced_features = ['victim_was_aware', 'had_sound_cue', 'utility_count', 'approach_align_deg']
        for feature in enhanced_features:
            if feature in sample_kill and sample_kill[feature] is not None:
                available_features.append(feature)
        
        for kill in labeled_data:
            features = []
            
            # Always include basic features with fallbacks
            features.append(float(kill.get('distance_xy', 0)))
            features.append(float(kill.get('time_in_round_s', 0)))
            features.append(1 if kill.get('headshot', False) else 0)
            
            # Add enhanced features if available, otherwise use defaults
            if 'victim_was_aware' in available_features:
                features.append(1 if kill.get('victim_was_aware', False) else 0)
            else:
                features.append(0)  # Default: not aware
            
            if 'had_sound_cue' in available_features:
                features.append(1 if kill.get('had_sound_cue', False) else 0)
            else:
                features.append(0)  # Default: no sound cue
            
            if 'utility_count' in available_features:
                features.append(float(kill.get('utility_count', 0)))
            else:
                features.append(0)  # Default: no utility
            
            if 'approach_align_deg' in available_features:
                features.append(float(kill.get('approach_align_deg', 0) or 0))
            else:
                features.append(0)  # Default: no movement
            
            # Use attacker labels as target
            if 'attacker_labels' in kill and kill['attacker_labels']:
                label = kill['attacker_labels'][0]
            else:
                label = 'other'
            
            training_features.append(features)
            training_labels.append(label)
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(training_labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            training_features, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Train model
        if use_lightgbm:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        val_accuracy = model.score(X_val, y_val)
        
        # If we have filtered_kills, calculate new uncertainties
        if filtered_kills is not None and not filtered_kills.empty:
            unlabeled_features = []
            unlabeled_indices = []
            
            for idx, kill_row in filtered_kills.iterrows():
                # Check if this kill is already labeled by comparing key attributes
                is_labeled = any(
                    (labeled_kill.get('tick') == kill_row.get('tick') and 
                     labeled_kill.get('attacker_name') == kill_row.get('attacker_name') and
                     labeled_kill.get('victim_name') == kill_row.get('victim_name'))
                    for labeled_kill in labeled_data
                )
                
                if not is_labeled:
                    # Extract features for unlabeled kill (same logic as training)
                    features = []
                    
                    # Always include basic features with fallbacks
                    features.append(float(kill_row.get('distance_xy', 0)))
                    features.append(float(kill_row.get('time_in_round_s', 0)))
                    features.append(1 if kill_row.get('headshot', False) else 0)
                    
                    # Add enhanced features if available, otherwise use defaults
                    if 'victim_was_aware' in available_features:
                        features.append(1 if kill_row.get('victim_was_aware', False) else 0)
                    else:
                        features.append(0)  # Default: not aware
                    
                    if 'had_sound_cue' in available_features:
                        features.append(1 if kill_row.get('had_sound_cue', False) else 0)
                    else:
                        features.append(0)  # Default: no sound cue
                    
                    if 'utility_count' in available_features:
                        features.append(float(kill_row.get('utility_count', 0)))
                    else:
                        features.append(0)  # Default: no utility
                    
                    if 'approach_align_deg' in available_features:
                        features.append(float(kill_row.get('approach_align_deg', 0) or 0))
                    else:
                        features.append(0)  # Default: no movement
                    
                    unlabeled_features.append(features)
                    unlabeled_indices.append(idx)
            
            if unlabeled_features:
                try:
                    # Get prediction probabilities
                    probabilities = model.predict_proba(unlabeled_features)
                    
                    # Calculate uncertainty (entropy)
                    uncertainties = []
                    for prob in probabilities:
                        # Add small epsilon to avoid log(0)
                        prob_safe = prob + 1e-10
                        prob_safe = prob_safe / prob_safe.sum()  # Renormalize
                        # Use log2 for entropy calculation (more standard)
                        entropy = -np.sum(prob_safe * np.log2(prob_safe + 1e-10))
                        uncertainties.append(entropy)
                    
                    # Sort by uncertainty (highest first)
                    uncertainty_data = list(zip(unlabeled_indices, uncertainties))
                    uncertainty_data.sort(key=lambda x: x[1], reverse=True)
                    
                    # Update session state with new suggestions
                    st.session_state['active_learning_suggestions'] = {
                        'suggested_indices': [idx for idx, _ in uncertainty_data[:10]],  # Default to 10
                        'uncertainties': [unc for _, unc in uncertainty_data[:10]],
                        'model_accuracy': val_accuracy,
                        'labeled_count': len(labeled_data),
                        'total_unlabeled': len(unlabeled_features)
                    }
                except Exception as e:
                    # If prediction fails, still update model info
                    st.session_state['active_learning_suggestions'] = {
                        'model_accuracy': val_accuracy,
                        'labeled_count': len(labeled_data),
                        'error': str(e)
                    }
        
        # Store model info even without filtered_kills
        if 'active_learning_suggestions' not in st.session_state:
            st.session_state['active_learning_suggestions'] = {
                'model_accuracy': val_accuracy,
                'labeled_count': len(labeled_data)
            }
        else:
            st.session_state['active_learning_suggestions']['model_accuracy'] = val_accuracy
            st.session_state['active_learning_suggestions']['labeled_count'] = len(labeled_data)
            
    except Exception as e:
        # Silently fail for auto-retrain to avoid disrupting the UI
        pass


def display_enhanced_kill_info(kill_context: Dict) -> None:
    """
    Display enhanced kill information with additional context.
    
    Args:
        kill_context: Enhanced kill context dictionary
    """
    st.subheader("🎯 Enhanced Kill Analysis")
    
    # Match and Round Information
    if kill_context.get('round_number') is not None:
        st.markdown("**🎮 Match & Round Info**")
        col_match, col_round = st.columns(2)
        
        with col_match:
            team1 = kill_context.get('team1_name', 'Team 1')
            team2 = kill_context.get('team2_name', 'Team 2')
            score_t = kill_context.get('match_score_t', 0)
            score_ct = kill_context.get('match_score_ct', 0)
            st.write(f"**Match Score:** {team1} {score_t} - {score_ct} {team2}")
        
        with col_round:
            round_num = kill_context.get('round_number', 0)
            round_phase = kill_context.get('round_phase', 'unknown')
            st.write(f"**Round:** {round_num} ({round_phase})")
            if 'bomb_planted' in kill_context and kill_context['bomb_planted']:
                st.write(f"**Bomb Planted:** Yes ({kill_context.get('time_since_bomb_plant', 0):.1f}s ago)")
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information:**")
        st.write(f"**Attacker:** {kill_context['attacker_name']}")
        st.write(f"**Victim:** {kill_context['victim_name']}")
        st.write(f"**Distance:** {kill_context['distance_xy']:.0f} units")
        st.write(f"**Headshot:** {'Yes' if kill_context['headshot'] else 'No'}")
        st.write(f"**Time in Round:** {kill_context['time_in_round_s']:.1f}s")
        
        # Weapon information
        st.markdown("**Weapons:**")
        attacker_weapon = kill_context.get('attacker_weapon', 'Unknown')
        victim_weapon = kill_context.get('victim_weapon', 'Unknown')
        st.write(f"**Attacker Weapon:** {attacker_weapon}")
        st.write(f"**Victim Weapon:** {victim_weapon}")
        
        # Equipment advantage analysis
        if attacker_weapon != 'Unknown' and victim_weapon != 'Unknown':
            st.markdown("**Equipment Analysis:**")
            # Simple weapon tier analysis
            primary_weapons = ['ak47', 'm4a1', 'awp', 'sg553', 'aug', 'famas', 'galil']
            secondary_weapons = ['deagle', 'usp', 'glock', 'p250', 'tec9', 'cz75']
            
            attacker_has_primary = any(weapon in attacker_weapon.lower() for weapon in primary_weapons)
            victim_has_primary = any(weapon in victim_weapon.lower() for weapon in primary_weapons)
            
            if attacker_has_primary and not victim_has_primary:
                st.write("✅ Attacker has equipment advantage")
            elif not attacker_has_primary and victim_has_primary:
                st.write("❌ Attacker has equipment disadvantage")
            else:
                st.write("⚖️ Equipment is balanced")
    
    with col2:
        st.markdown("**Player States:**")
        st.write(f"**Attacker Health:** {kill_context.get('attacker_health', 100)}")
        st.write(f"**Victim Health:** {kill_context.get('victim_health', 100)}")
        st.write(f"**Attacker Moving:** {'Yes' if kill_context.get('attacker_is_moving', False) else 'No'}")
        st.write(f"**Victim Moving:** {'Yes' if kill_context.get('victim_is_moving', False) else 'No'}")
        st.write(f"**Attacker Ducking:** {'Yes' if kill_context.get('attacker_is_ducking', False) else 'No'}")
        st.write(f"**Victim Ducking:** {'Yes' if kill_context.get('victim_is_ducking', False) else 'No'}")
        
        # Approach alignment with interpretation
        approach_align = kill_context.get('approach_align_deg')
        if approach_align is not None:
            st.write(f"**Approach Alignment:** {approach_align:.1f}°")
            if approach_align < 30:
                st.write("🎯 Excellent alignment")
            elif approach_align < 60:
                st.write("✅ Good alignment")
            elif approach_align < 90:
                st.write("⚠️ Fair alignment")
            else:
                st.write("❌ Poor alignment")
        else:
            st.write("**Approach Alignment:** Not moving")
    
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
