#!/usr/bin/env python3
"""
Tacticore - CS2 Kill Event Labeling App

A Streamlit application for labeling Counter-Strike 2 kill events with
attacker and victim context, map overlays, and tactical analysis.
Enhanced version with batch labeling and ML training features.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components import (
    create_filters, apply_filters, create_label_controls, create_map_figure,
    display_kill_info, create_navigation_controls, display_labeled_summary,
    create_export_button, create_file_uploaders, create_map_settings,
    create_batch_labeling_controls, create_ml_training_controls, create_labeled_data_importer,
    create_active_learning_navigation, create_labeled_data_display,
    display_model_predictions, create_live_labeling_feedback, create_enhanced_active_learning_navigation,
    create_model_simulation_mode, show_incremental_training_guide
)
from transforms import (
    load_map_data, get_enhanced_kill_context
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
    .batch-mode {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
    }
    .ml-mode {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
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
    
    if 'batch_mode' not in st.session_state:
        st.session_state.batch_mode = False
    
    if 'ml_mode' not in st.session_state:
        st.session_state.ml_mode = False
    
    if 'simulation_mode' not in st.session_state:
        st.session_state.simulation_mode = False
    
    if 'batch_labels' not in st.session_state:
        st.session_state.batch_labels = {}


def get_active_learning_status(current_index: int, total_kills: int) -> Tuple[bool, str]:
    """
    Check if current kill is prioritized by active learning.
    
    Args:
        current_index: Current kill index
        total_kills: Total number of kills
        
    Returns:
        Tuple of (is_active_learning, display_text)
    """
    if 'active_learning_suggestions' in st.session_state:
        suggestions = st.session_state['active_learning_suggestions']
        if suggestions and 'suggested_indices' in suggestions:
            if current_index in suggestions['suggested_indices']:
                uncertainty_idx = suggestions['suggested_indices'].index(current_index)
                uncertainty = suggestions['uncertainties'][uncertainty_idx]
                accuracy = suggestions.get('model_accuracy', 0)
                return True, f'🎯 **ACTIVE LEARNING** - Kill {current_index + 1} of {total_kills} (Uncertainty: {uncertainty:.3f}, Model Acc: {accuracy:.2f})'
    
    return False, f'Kill {current_index + 1} of {total_kills}'


def save_labeled_kill(kill_context: Dict, attacker_labels: List[str], victim_labels: List[str]) -> None:
    """
    Save a labeled kill to session state.
    
    Args:
        kill_context: Dictionary with kill context
        attacker_labels: List of labels for attacker
        victim_labels: List of labels for victim
    """
    labeled_kill = kill_context.copy()
    labeled_kill['attacker_labels'] = attacker_labels
    labeled_kill['victim_labels'] = victim_labels
    labeled_kill['attacker_label'] = ', '.join(attacker_labels) if attacker_labels else ''
    labeled_kill['victim_label'] = ', '.join(victim_labels) if victim_labels else ''
    
    existing_indices = [
        i for i, labeled in enumerate(st.session_state.labeled_data)
        if (labeled.get('kill_tick') == kill_context.get('kill_tick') and
            labeled.get('attacker_name') == kill_context.get('attacker_name') and
            labeled.get('victim_name') == kill_context.get('victim_name'))
    ]
    
    if existing_indices:
        st.session_state.labeled_data[existing_indices[0]] = labeled_kill
        st.success("Updated existing label")
    else:
        st.session_state.labeled_data.append(labeled_kill)
        st.success("Label saved!")
    
    st.session_state.current_attacker_labels = []
    st.session_state.current_victim_labels = []
    
    total_kills = len(st.session_state.filtered_kills) if hasattr(st.session_state, 'filtered_kills') else 0
    if total_kills > 0:
        next_index = min(st.session_state.current_kill_index + 1, total_kills - 1)
        st.session_state.current_kill_index = next_index
        st.success(f"✅ Labels saved! Advanced to kill {next_index + 1} of {total_kills}")
    else:
        st.success("✅ Labels saved!")
    
    st.info("🔄 Use 'Train Model' button in sidebar to retrain when you want.")


def save_batch_labels(batch_labels: Dict, kill_contexts: List[Dict]) -> None:
    """
    Save multiple labeled kills from batch mode.
    
    Args:
        batch_labels: Dictionary mapping kill indices to labels
        kill_contexts: List of kill contexts
    """
    saved_count = 0
    
    for kill_idx, labels in batch_labels.items():
        if kill_idx < len(kill_contexts):
            kill_context = kill_contexts[kill_idx]
            attacker_labels = labels.get('attacker', [])
            victim_labels = labels.get('victim', [])
            
            if attacker_labels or victim_labels:
                save_labeled_kill(kill_context, attacker_labels, victim_labels)
                saved_count += 1
    
    if saved_count > 0:
        st.success(f"Saved {saved_count} batch labels!")
        st.session_state.batch_labels = {}


def main():
    """Main application function."""
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🎯 Tacticore - CS2 Kill Labeling</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("📊 Tacticore")
    st.sidebar.markdown("Label CS2 kill events with tactical context")
    
    mode = st.sidebar.selectbox(
        "Labeling Mode",
        ["Single Kill", "Batch Mode", "ML Training Mode", "Model Simulation"],
        index=0
    )
    
    if mode == "Batch Mode":
        st.session_state.batch_mode = True
        st.session_state.ml_mode = False
        st.session_state.simulation_mode = False
    elif mode == "ML Training Mode":
        st.session_state.batch_mode = False
        st.session_state.ml_mode = True
        st.session_state.simulation_mode = False
    elif mode == "Model Simulation":
        st.session_state.batch_mode = False
        st.session_state.ml_mode = False
        st.session_state.simulation_mode = True
    else:
        st.session_state.batch_mode = False
        st.session_state.ml_mode = False
        st.session_state.simulation_mode = False
    
    kills_df, ticks_df, grenades_df, damages_df, shots_df, smokes_df, infernos_df, bomb_df, rounds_df = create_file_uploaders()
    
    create_labeled_data_importer()
    
    if st.session_state.ml_mode and (damages_df is None or shots_df is None or smokes_df is None or infernos_df is None or bomb_df is None):
        st.sidebar.markdown("### Additional Data Files (ML Mode)")
        
        if damages_df is None:
            damages_df = st.sidebar.file_uploader("Damage Events (damages.parquet)", type=['parquet'])
        if shots_df is None:
            shots_df = st.sidebar.file_uploader("Shot Events (shots.parquet)", type=['parquet'])
        if smokes_df is None:
            smokes_df = st.sidebar.file_uploader("Smoke Events (smokes.parquet)", type=['parquet'])
        if infernos_df is None:
            infernos_df = st.sidebar.file_uploader("Molotov Events (infernos.parquet)", type=['parquet'])
        if bomb_df is None:
            bomb_df = st.sidebar.file_uploader("Bomb Events (bomb.parquet)", type=['parquet'])
    
    if kills_df is None or ticks_df is None:
        st.warning("Please upload both kills.parquet and ticks.parquet files to continue.")
        return
    
    map_image_path, map_data_path, tickrate, figure_size, x_fine_tune, y_fine_tune, use_advanced = create_map_settings()
    
    selected_map = st.session_state.get("map_selection", "de_mirage")
    
    filters = create_filters(kills_df)
    filtered_kills = apply_filters(kills_df, filters)
    st.session_state.filtered_kills = filtered_kills
    st.session_state.total_kills_estimate = len(filtered_kills)
    
    map_data = {}
    if Path(map_data_path).exists():
        map_data = load_map_data(map_data_path, selected_map)
        
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
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Kills:** {len(filtered_kills)} / {len(kills_df)}")
    
    if len(filtered_kills) == 0:
        st.warning("No kills match the current filters. Please adjust your filters.")
        return
    
    damages_df_loaded = damages_df if damages_df is not None else pd.DataFrame()
    shots_df_loaded = shots_df if shots_df is not None else pd.DataFrame()
    smokes_df_loaded = smokes_df if smokes_df is not None else pd.DataFrame()
    infernos_df_loaded = infernos_df if infernos_df is not None else pd.DataFrame()
    bomb_df_loaded = bomb_df if bomb_df is not None else pd.DataFrame()
    
    if st.session_state.batch_mode:
        st.markdown('<div class="batch-mode">', unsafe_allow_html=True)
        st.header("📦 Batch Labeling Mode")
        st.markdown("Label multiple kills at once for faster processing")
        st.markdown('</div>', unsafe_allow_html=True)
        
        batch_size = st.slider("Batch Size", 5, 20, 10)
        start_index = st.slider("Start Index", 0, len(filtered_kills) - batch_size, 0)
        
        end_index = min(start_index + batch_size, len(filtered_kills))
        batch_kills = filtered_kills.iloc[start_index:end_index]
        
        kill_contexts = []
        
        if rounds_df is None or rounds_df.empty:
            rounds_df = pd.DataFrame()
            if not ticks_df.empty and 'tick' in ticks_df.columns:
                min_tick = ticks_df['tick'].min()
                max_tick = ticks_df['tick'].max()
                round_duration = 3000
                estimated_rounds = []
                for round_num in range(1, 31):
                    start_tick = min_tick + (round_num - 1) * round_duration
                    end_tick = min_tick + round_num * round_duration
                    if start_tick <= max_tick:
                        estimated_rounds.append({
                            'round': round_num,
                            'start_tick': start_tick,
                            'end_tick': end_tick
                        })
                if estimated_rounds:
                    rounds_df = pd.DataFrame(estimated_rounds)
        
        for _, kill_row in batch_kills.iterrows():
            if st.session_state.ml_mode:
                context = get_enhanced_kill_context(
                    kill_row, ticks_df, rounds_df, grenades_df if grenades_df is not None else pd.DataFrame(),
                    damages_df_loaded if damages_df_loaded is not None else pd.DataFrame(),
                    shots_df_loaded if shots_df_loaded is not None else pd.DataFrame(),
                    smokes_df_loaded if smokes_df_loaded is not None else pd.DataFrame(),
                    infernos_df_loaded if infernos_df_loaded is not None else pd.DataFrame(),
                    bomb_df_loaded if bomb_df_loaded is not None else pd.DataFrame(),
                    map_data, tickrate, x_fine_tune, y_fine_tune, use_advanced
                )
            else:
                context = get_enhanced_kill_context(
                    kill_row, ticks_df, rounds_df, grenades_df if grenades_df is not None else pd.DataFrame(),
                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                    map_data, tickrate, x_fine_tune, y_fine_tune, use_advanced
                )
            kill_contexts.append(context)
        
        batch_labels = create_batch_labeling_controls(batch_kills, kill_contexts, map_data, map_image_path, figure_size)
        
        if st.button("💾 Save Batch Labels", type="primary"):
            save_batch_labels(batch_labels, kill_contexts)
            st.info("🔄 Batch labels saved. Click 'Refresh Display' to see updated data.")
    
    elif st.session_state.ml_mode:
        st.markdown('<div class="ml-mode">', unsafe_allow_html=True)
        st.header("🤖 ML Training Mode")
        st.markdown("Enhanced labeling with ML assistance and active learning")
        st.markdown('</div>', unsafe_allow_html=True)
        
        ml_controls = create_ml_training_controls(filtered_kills, st.session_state.labeled_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎯 Enhanced Kill Analysis")
            
            current_index = st.session_state.current_kill_index
            total_kills = len(filtered_kills)
            
            is_active_learning, display_text = get_active_learning_status(current_index, total_kills)
            st.markdown(f'<div class="kill-counter">{display_text}</div>', 
                       unsafe_allow_html=True)
            
            current_kill = filtered_kills.iloc[current_index]
            
            if rounds_df is None or rounds_df.empty:
                rounds_df = pd.DataFrame()
                if not ticks_df.empty and 'tick' in ticks_df.columns:
                    min_tick = ticks_df['tick'].min()
                    max_tick = ticks_df['tick'].max()
                    round_duration = 3000
                    estimated_rounds = []
                    for round_num in range(1, 31):
                        start_tick = min_tick + (round_num - 1) * round_duration
                        end_tick = min_tick + round_num * round_duration
                        if start_tick <= max_tick:
                            estimated_rounds.append({
                                'round': round_num,
                                'start_tick': start_tick,
                                'end_tick': end_tick
                            })
                    if estimated_rounds:
                        rounds_df = pd.DataFrame(estimated_rounds)
            
            kill_context = get_enhanced_kill_context(
                current_kill, ticks_df, rounds_df, grenades_df if grenades_df is not None else pd.DataFrame(),
                damages_df_loaded if damages_df_loaded is not None else pd.DataFrame(),
                shots_df_loaded if shots_df_loaded is not None else pd.DataFrame(),
                smokes_df_loaded if smokes_df_loaded is not None else pd.DataFrame(),
                infernos_df_loaded if infernos_df_loaded is not None else pd.DataFrame(),
                bomb_df_loaded if bomb_df_loaded is not None else pd.DataFrame(),
                map_data, tickrate, x_fine_tune, y_fine_tune, use_advanced
            )
            
            display_kill_info(kill_context)
            
            if st.checkbox("🔍 Debug: Show Player Names", value=False):
                st.write("**All unique attacker names:**")
                if 'kills_df' in st.session_state and not st.session_state.kills_df.empty:
                    attacker_names = st.session_state.kills_df['attacker_name'].unique()
                    st.write(attacker_names)
                    underscore_attackers = [name for name in attacker_names if str(name).startswith('_')]
                    if underscore_attackers:
                        st.write(f"**Names starting with underscore:** {underscore_attackers}")
                    else:
                        st.write("**No names starting with underscore found**")
                st.write("**All unique victim names:**")
                if 'kills_df' in st.session_state and not st.session_state.kills_df.empty:
                    victim_names = st.session_state.kills_df['victim_name'].unique()
                    st.write(victim_names)
                    underscore_victims = [name for name in victim_names if str(name).startswith('_')]
                    if underscore_victims:
                        st.write(f"**Names starting with underscore:** {underscore_victims}")
                    else:
                        st.write("**No names starting with underscore found**")
            
            display_model_predictions(kill_context, current_index, filtered_kills)
            
            if map_data and Path(map_image_path).exists():
                st.subheader("🗺️ Map Location")
                
                attacker_pos = (kill_context['attacker_image_x'], kill_context['attacker_image_y'])
                victim_pos = (kill_context['victim_image_x'], kill_context['victim_image_y'])
                
                map_fig = create_map_figure(map_image_path, map_data, attacker_pos, victim_pos, figure_size,
                                           kill_context['attacker_name'], kill_context['victim_name'])
                if map_fig:
                    st.pyplot(map_fig)
            else:
                st.warning("Map image or data not available")
        
        with col2:
            st.header("🏷️ Enhanced Labeling")
            
            attacker_labels, victim_labels = create_label_controls()
            
            if st.button("💾 Save Labels", type="primary"):
                if attacker_labels or victim_labels:
                    save_labeled_kill(kill_context, attacker_labels, victim_labels)
                    
                    st.session_state.current_attacker_labels = []
                    st.session_state.current_victim_labels = []
                    
                    st.info("✅ Labels saved! Use navigation buttons to go to next kill.")
                else:
                    st.warning("Please select at least one label")
            
            create_active_learning_navigation(current_index, total_kills, is_active_learning)
            
            new_index = create_navigation_controls(total_kills, current_index)
            if new_index != current_index:
                st.session_state.current_kill_index = new_index
                st.session_state.current_attacker_labels = []
                st.session_state.current_victim_labels = []
    
    elif st.session_state.simulation_mode:
        st.markdown('<div class="ml-mode">', unsafe_allow_html=True)
        st.header("🤖 Model Simulation Mode")
        st.markdown("Test the model's ability to predict labels and correct it to improve learning")
        st.markdown('</div>', unsafe_allow_html=True)
        
        labeled_kills_df = pd.DataFrame(st.session_state.labeled_data) if st.session_state.labeled_data else pd.DataFrame()
        model_predictions = st.session_state.get('model_predictions', {})
        
        create_model_simulation_mode(
            filtered_kills, 
            labeled_kills_df, 
            model_predictions, 
            map_data, 
            tickrate, 
            x_fine_tune, 
            y_fine_tune, 
            use_advanced
        )
    
    else:
        # Standard single kill mode
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎯 Kill Analysis")
            
            current_index = st.session_state.current_kill_index
            total_kills = len(filtered_kills)
            
            is_active_learning, display_text = get_active_learning_status(current_index, total_kills)
            st.markdown(f'<div class="kill-counter">{display_text}</div>', 
                       unsafe_allow_html=True)
            
            current_kill = filtered_kills.iloc[current_index]
            
            if rounds_df is None or rounds_df.empty:
                rounds_df = pd.DataFrame()
                if not ticks_df.empty and 'tick' in ticks_df.columns:
                    min_tick = ticks_df['tick'].min()
                    max_tick = ticks_df['tick'].max()
                    round_duration = 3000
                    estimated_rounds = []
                    for round_num in range(1, 31):
                        start_tick = min_tick + (round_num - 1) * round_duration
                        end_tick = min_tick + round_num * round_duration
                        if start_tick <= max_tick:
                            estimated_rounds.append({
                                'round': round_num,
                                'start_tick': start_tick,
                                'end_tick': end_tick
                            })
                    if estimated_rounds:
                        rounds_df = pd.DataFrame(estimated_rounds)
            
            kill_context = get_enhanced_kill_context(
                current_kill, ticks_df, rounds_df, grenades_df if grenades_df is not None else pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                map_data, tickrate, x_fine_tune, y_fine_tune, use_advanced
            )
            
            display_kill_info(kill_context)
            
            if map_data and Path(map_image_path).exists():
                st.subheader("🗺️ Map Location")
                
                attacker_pos = (kill_context['attacker_image_x'], kill_context['attacker_image_y'])
                victim_pos = (kill_context['victim_image_x'], kill_context['victim_image_y'])
                
                map_fig = create_map_figure(map_image_path, map_data, attacker_pos, victim_pos, figure_size,
                                           kill_context['attacker_name'], kill_context['victim_name'])
                if map_fig:
                    st.pyplot(map_fig)
            else:
                st.warning("Map image or data not available")
        
        with col2:
            st.header("🏷️ Labeling")
            
            attacker_labels, victim_labels = create_label_controls()
            
            if st.button("💾 Save Labels", type="primary"):
                if attacker_labels or victim_labels:
                    save_labeled_kill(kill_context, attacker_labels, victim_labels)
                    
                    st.session_state.current_attacker_labels = []
                    st.session_state.current_victim_labels = []
                    
                    st.info("✅ Labels saved! Use navigation buttons to go to next kill.")
                else:
                    st.warning("Please select at least one label")
            
            create_active_learning_navigation(current_index, total_kills, is_active_learning)
            
            new_index = create_navigation_controls(total_kills, current_index)
            if new_index != current_index:
                st.session_state.current_kill_index = new_index
                st.session_state.current_attacker_labels = []
                st.session_state.current_victim_labels = []
    
    create_live_labeling_feedback()
    create_labeled_data_display()
    
    if st.session_state.labeled_data and len(st.session_state.labeled_data) > 0:
        show_incremental_training_guide()
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use Tacticore:
        
        1. **Upload Data**: Upload your parsed demo files (kills.parquet, ticks.parquet, optional grenades.parquet)
        2. **Configure Map**: Set the map image path and map data JSON file
        3. **Choose Mode**: Select Single Kill, Batch Mode, ML Training Mode, or Model Simulation
        4. **Filter Kills**: Use the sidebar filters to focus on specific kills
        5. **Analyze Context**: Review the kill information and map location
        6. **Apply Labels**: Select appropriate labels for both attacker and victim
        7. **Save & Navigate**: Save labels and manually navigate to the next kill
        8. **Export Results**: Download your labeled dataset when finished
        
        ### Labeling Modes:
        
        **Single Kill Mode**: Traditional one-by-one labeling
        **Batch Mode**: Label multiple kills at once for faster processing
        **ML Training Mode**: Enhanced context with ML assistance and active learning
        **Model Simulation**: Test the model's predictions and correct them to improve learning
        
        ### Enhanced Context Features (ML Mode):
        
        - **Sound Cues**: Detect if victim heard attacker before death
        - **Victim Awareness**: Detect if victim was watching the attacker or was backstabbed
        - **Round Context**: Time in round, bomb status, round phase
        - **Utility Analysis**: Active grenades, smokes, molotovs
        - **Player States**: Movement, weapons, positioning
        - **Tactical Analysis**: Distance categories, advantages
        
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
        - `bad_clearing` - Victim failed to clear an area properly
        - `other` - Other factors
        
        **Multiple Labels:** You can now select multiple labels for both attacker and victim to capture complex situations where a player has both good and bad aspects to their play.
        """)


if __name__ == "__main__":
    main()
