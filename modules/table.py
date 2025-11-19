import pandas as pd
import re
import os
import jellyfish as jf

def excel_styler(df: pd.DataFrame):
    styled_df = df.style.map_index(
        lambda x: "background-color: #000000; color: #fff; font-family: Arial; font-size: 9pt; font-weight: bold; text-align: center;",
        axis=1,
    )
    styled_df = styled_df.set_properties(
        **{
            "font-family": "Arial",
            "font-size": "9pt",
            "text-align": "left",
            "border": "1px solid black",
        }
    )
    return styled_df

def pivot_table(df: pd.DataFrame, index: str, columns: str, values: str, aggfunc='first') -> pd.DataFrame:
    pivot_df = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
    pivot_df = pivot_df.reset_index()
    pivot_df.columns.name = None
    return pivot_df

def find_best_match(word, candidates, threshold=0.85):
    best_match = word
    best_score = 0
    for candidate in candidates:
        score = jf.jaro_winkler_similarity(word, candidate)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    return best_match, best_score

def sanitize_header(df, preview_row = 5, lowercase=False):
    if df.columns[0].startswith('Unnamed'):
        for idx, row in df.head(preview_row).iterrows():
            if pd.isna(row.values[0]) or row.values[0] == '':
                continue
            else:
                df.columns = [str(col).strip() for col in row]
                df = df.iloc[idx + 1:].reset_index(drop=True)
                print(f"Header sanitized | Start from row {idx + 1} | Columns: {[col.strip() for col in df.columns[:3]]} ...")
                break
    else:
        df.columns = [col.strip() for col in df.columns]
    
    # Drop nan columns
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    df = df.loc[:, ~df.columns.str.contains('nan', na=False)]
    
    # Clean entered columns
    df.columns = [col.split('.')[0] for col in df.columns]
    df.columns = [col.strip().replace('\n', '') for col in df.columns]
    df.columns = df.columns.str.replace("*", "")

    if lowercase:
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    
    # Clean duplicate columns
    duplicated_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicated_cols:
        print(f"‼️ Duplicate columns found")
        new_columns = []
        col_count = {}
        for col in df.columns:
            if col in duplicated_cols:
                col_count[col] = col_count.get(col, 0) + 1
                new_col_name = f"{col}_{col_count[col]}"
                new_columns.append(new_col_name)
            else:
                new_columns.append(col)
        print(f"‼️ Duplicate columns found")
        for col in duplicated_cols:
            print(f"  - {col} | Total: {col_count[col]}")
        df.columns = new_columns
    else:
        print("No duplicate columns found.")
    return df

def detect_week(date_str):
    try:
        date_obj = pd.to_datetime(date_str, format='%Y%m%d')
        week_number = date_obj.isocalendar()[1]
        return week_number
    except ValueError:
        print(f"❌ Invalid date format: {date_str}")
        return None

def detect_version(filepath: str | os.PathLike) -> str:
    filename = str(filepath).lower().split(os.sep)[-1]
    print(f"Detecting version from filename: {filename}")
    search_version = re.search(r'v(?P<version>\d+)', filename)
    if search_version:
        version = search_version.group('version')
        new_version = f"v{int(version) + 1}"
        print(f"Version detected: {version} | New version will be: {new_version}")
        return new_version
    else:
        print("No version detected in the filename.")
        return "v1"