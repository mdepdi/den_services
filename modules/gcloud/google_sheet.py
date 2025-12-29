from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Sequence

import pandas as pd
import gspread
from gspread.client import Client
from gspread.spreadsheet import Spreadsheet
from gspread.worksheet import Worksheet
from google.oauth2.service_account import Credentials

DEFAULT_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
)

class SpreadsheetError(RuntimeError):
    """Raised when spreadsheet operations fail."""


def authenticate_credentials(
    creds_path: Union[str, Path],
    scopes: Sequence[str] = DEFAULT_SCOPES,
    *,
    logger=None,
) -> Client:
    """
    Authenticate using a Google service account JSON file and return a gspread Client.

    Args:
        creds_path: Path to service account JSON credentials.
        scopes: OAuth scopes to request.
        logger: Optional logger with .info().

    Returns:
        gspread Client.
    """
    path = Path(creds_path)
    if path.suffix.lower() != ".json":
        raise ValueError("Credentials file must be a .json file.")

    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    try:
        import json
        with path.open("r", encoding="utf-8") as f:
            cred_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read credentials JSON: {e}") from e

    email = cred_data.get("client_email")
    if not email:
        raise ValueError("Invalid credentials JSON: 'client_email' not found.")

    credentials = Credentials.from_service_account_file(
        filename=str(path),
        scopes=list(scopes),
    )
    gc = gspread.authorize(credentials)

    print(f"🟢 Credentials granted for {email}")
    return gc


def load_spreadsheet(gc: Client, kut: str) -> Spreadsheet:
    """
    Open a spreadsheet by key, URL, or title (in that order).
    """
    # Key
    try:
        return gc.open_by_key(kut)
    except Exception:
        pass

    # URL
    try:
        return gc.open_by_url(kut)
    except Exception:
        pass

    # Title
    try:
        return gc.open(kut)
    except Exception as e:
        raise SpreadsheetError(f"Unable to open spreadsheet '{kut}': {e}") from e


def get_worksheet(sheet: Spreadsheet, sheet_name: Union[int, str, None]) -> Worksheet:
    if sheet_name is None:
        return sheet.get_worksheet(0)

    if isinstance(sheet_name, int):
        ws = sheet.get_worksheet(sheet_name)
        if ws is None:
            raise ValueError(f"Worksheet index out of range: {sheet_name}")
        return ws

    if isinstance(sheet_name, str):
        return sheet.worksheet(sheet_name)
    raise TypeError(f"sheet_name must be int, str, or None; got {type(sheet_name)}")


def open_spreadsheet(
    key_or_url_or_title: str,
    sheet_name: Union[int, str, None] = None,
    *,
    gc: Optional[Client] = None,
    creds_path: Union[str, Path] = "creds_service_wso.json",
) -> pd.DataFrame:
    """
    Open a spreadsheet and return worksheet records as a DataFrame.
    """
    if gc is None:
        gc = authenticate_credentials(creds_path)

    sheet = load_spreadsheet(gc, key_or_url_or_title)
    ws = get_worksheet(sheet, sheet_name)

    records = ws.get_all_records()
    records_df = pd.DataFrame(records)
    print(f"🟢 Dataframe fetched with {len(records_df):,} rows from spreadsheet.")
    return records_df

def build_records(df: pd.DataFrame) -> list[list[object]]:
    """
    Convert a DataFrame into a 2D array suitable for gspread Worksheet.update().
    Includes header row.
    """
    header = df.columns.astype(str).tolist()
    body = df.where(pd.notnull(df), "").values.tolist()  # replace NaN with empty string
    return [header] + body


def write_spreadsheet(
    data: pd.DataFrame,
    *,
    title: str,
    folder_id: Optional[str] = None,
    key: Optional[str] = None,
    sheet_name: Union[str, int, None] = None,
    gc: Optional[Client] = None,
    creds_path: Union[str, Path] = "creds_service_wso.json",
) -> Spreadsheet:
    """
    Create a new spreadsheet or update an existing one with DataFrame contents.

    Args:
        data: DataFrame to write.
        title: Spreadsheet title (used when creating new).
        folder_id: Drive folder id for creation (optional).
        key: If provided, updates that spreadsheet; otherwise creates.
        sheet_name: Worksheet index/name (defaults to first worksheet).
        gc: Optional authenticated gspread client.
        creds_path: Used if gc is not provided.
        clear_before_write: Clear sheet before writing (recommended).

    Returns:
        The Spreadsheet object.
    """
    if gc is None:
        gc = authenticate_credentials(creds_path)

    if data is None or data.empty:
        raise ValueError("DataFrame is empty. Nothing to write.")

    values = build_records(data)

    # Create
    if not key:
        sheet = gc.create(title, folder_id=folder_id)
        ws = sheet.get_worksheet(0)

        if isinstance(sheet_name, str) and sheet_name.strip():
            ws.update_title(sheet_name.strip())

        ws.clear()
        ws.update(values)

        print(f"🟢 Spreadsheet created: {sheet.url}")
        return sheet

    # Update existing
    sheet = load_spreadsheet(gc, key)
    ws = get_worksheet(sheet, sheet_name)

    ws.clear()
    ws.update(values)

    print(f"🟢 Spreadsheet updated: {sheet.title} with url {sheet.url})")
    return sheet

if __name__ == "__main__":
    spreadsheet_key = "1AFUF7G6Wv_YEaGusuFIAn82Px_oO4HgM2OLmj_z3jT0"
    sitelist_automation = open_spreadsheet(spreadsheet_key)
    sitelist_automation['Creator'] = "Yakub H"
    write_spreadsheet(sitelist_automation, title="Trial Spreadsheet Automate", key=spreadsheet_key)