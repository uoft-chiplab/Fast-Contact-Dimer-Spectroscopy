#!/usr/bin/env python3
"""
Excel to CSV Batch Converter
Converts all Excel files (.xlsx, .xls) in a directory to CSV format.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional

def find_excel_files(directory: str) -> List[Path]:
    """Find all Excel files in the specified directory."""
    directory_path = Path(directory)
    excel_extensions = ['.xlsx', '.xls', '.xlsm']
    excel_files = []
    
    for ext in excel_extensions:
        excel_files.extend(directory_path.glob(f'*{ext}'))
        excel_files.extend(directory_path.glob(f'*{ext.upper()}'))
    
    return sorted(excel_files)

def convert_excel_to_csv(excel_file: Path, output_dir: Optional[str] = None, 
                        sheet_name: Optional[str] = None) -> bool:
    """Convert a single Excel file to CSV format."""
    try:
        # Determine output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = excel_file.parent
        
        # Read Excel file
        if sheet_name:
            # Convert specific sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            csv_filename = f"{excel_file.stem}_{sheet_name}.csv"
        else:
            # Check if file has multiple sheets
            xl_file = pd.ExcelFile(excel_file)
            
            if len(xl_file.sheet_names) == 1:
                # Single sheet - simple conversion
                df = pd.read_excel(excel_file)
                csv_filename = f"{excel_file.stem}.csv"
            else:
                # Multiple sheets - convert each sheet
                for sheet in xl_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet)
                    csv_filename = f"{excel_file.stem}_{sheet}.csv"
                    csv_path = output_path / csv_filename
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    print(f"✓ Converted sheet '{sheet}' to: {csv_path}")
                return True
        
        # Save to CSV
        csv_path = output_path / csv_filename
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ Converted: {excel_file.name} → {csv_filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {excel_file.name}: {str(e)}")
        return False

def batch_convert(input_dir: str, output_dir: Optional[str] = None, 
                 sheet_name: Optional[str] = None) -> None:
    """Batch convert all Excel files in a directory."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        return
    
    # Find Excel files
    excel_files = find_excel_files(input_dir)
    
    if not excel_files:
        print(f"No Excel files found in '{input_dir}'")
        return
    
    print(f"Found {len(excel_files)} Excel file(s) to convert:")
    for file in excel_files:
        print(f"  - {file.name}")
    print()
    
    # Convert files
    successful = 0
    failed = 0
    
    for excel_file in excel_files:
        if convert_excel_to_csv(excel_file, output_dir, sheet_name):
            successful += 1
        else:
            failed += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":

    input_dir = "/Users/maggie/Documents/Chip lab analysis/Fast-Contact-Dimer-Spectroscopy/Summary/Manuscript_Figs" #Directory containing Excel files to convert
    output = "./" #Output directory for CSV files (default: same as input)
    sheet = "" #Specific sheet name to convert (default: all sheets)

    batch_convert(input_dir)

