#!/usr/bin/env python3
"""
Streamlit interface para parsear demos y generar .parquet files
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from parser.parse_dem_to_parquet import parse_single_demo

def main():
    st.set_page_config(
        page_title="Tacticore Demo Parser",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Tacticore Demo Parser")
    st.markdown("Parse demo files and generate .parquet files for labeling")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a .dem file",
        type=['dem'],
        help="Upload a Counter-Strike 2 demo file (.dem)"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size / (1024*1024):.1f} MB")
        
        # Output directory selection
        output_dir = st.text_input(
            "Output directory",
            value="dataset",
            help="Directory where .parquet files will be saved"
        )
        
        if st.button("Parse Demo", type="primary"):
            with st.spinner("Parsing demo file..."):
                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.dem') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Parse the demo
                    result = parse_single_demo(tmp_path, output_dir=output_dir)
                    
                    if result:
                        st.success("Demo parsed successfully!")
                        
                        # Show generated files
                        output_path = Path(output_dir)
                        if output_path.exists():
                            st.subheader("Generated Files:")
                            
                            # List .parquet files
                            parquet_files = list(output_path.rglob("*.parquet"))
                            if parquet_files:
                                st.write("**Parquet Files:**")
                                for file_path in parquet_files:
                                    st.write(f"  - {file_path.relative_to(output_path)}")
                            
                            # Check for meta.json
                            meta_file = output_path / "meta.json"
                            if meta_file.exists():
                                st.write("**Metadata:**")
                                st.write(f"  - meta.json")
                            
                            st.success("You can now use these files in the main Streamlit app for labeling!")
                        else:
                            st.error("Output directory not found")
                    else:
                        st.error("Failed to parse demo file")
                    
                    # Clean up temporary file
                    Path(tmp_path).unlink(missing_ok=True)
                    
                except Exception as e:
                    st.error(f"Error parsing demo: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Instructions
    st.markdown("---")
    st.markdown("### Instructions:")
    st.markdown("""
    1. Upload a .dem file using the file uploader above
    2. Specify the output directory (default: dataset)
    3. Click "Parse Demo" to generate .parquet files
    4. Use the generated files in the main Streamlit app for labeling
    """)
    
    # Show existing datasets
    st.markdown("### Existing Datasets:")
    dataset_dir = Path("dataset")
    if dataset_dir.exists():
        datasets = [d for d in dataset_dir.iterdir() if d.is_dir()]
        if datasets:
            for dataset in datasets:
                st.write(f"📁 {dataset.name}")
        else:
            st.info("No datasets found. Parse a demo to create one.")
    else:
        st.info("No dataset directory found. Parse a demo to create one.")

if __name__ == "__main__":
    main()
