import pandas as pd
from pathlib import Path
from typing import List, Optional
import pickle

def find_files(dir:str):
    dir_path = Path(dir)
    files = []
    files.extend(dir_path.glob(f'*pkl'))

    return sorted(files)

def convert_pkl_csv(fpath, output_dir=None):
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = fpath.parent

    csv_name = f"{fpath.stem}.csv"
    csv_path = output_path/csv_name

    data = pd.read_pickle(fpath)
    
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    return True

def batch_convert(input_dir, output_dir=None):
    input_path = Path(input_dir)

    files = find_files(input_dir)
    for f in files:
        try:
            convert_pkl_csv(f, output_dir)
            print(f" - {f.name} success")
        except:
            print(print(f" - {f.name} failed"))

if __name__ == "__main__":
    batch_convert(r"clockshift/rf_saturation_analysis/saturation_data", "data/")