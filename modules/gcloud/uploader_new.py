import os
import mimetypes
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
from tqdm import tqdm
import datetime

# PARAMETERS
FOLDER_PATH = r"H:\Orendo\TSEL Capture\Invalid No Streetview 903\Photo Result"
DRIVE_ID = "195I52G8OWy3JF3wDkJLNIdGYi9M1LfnM"

# FOLDER_PATH = r"H:\Orendo\TSEL Capture\20250725 - List HP INVALID 124K (Part of 903K CLEAN)\Photo Result"
# DRIVE_ID = "1QrHBtxk2L7Dgy0PybCfX-D9PAs-JBJ6-"

# Drive Configuration
SCOPES = ["https://www.googleapis.com/auth/drive"]
# SCOPES = ['https://www.googleapis.com/auth/drive.file']
# SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

TOKEN_DIR = r"H:\Orendo\Uploader\secrets"


# MODULES
def authenticate_credentials():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    if os.path.exists(os.path.join(TOKEN_DIR, "token.json")):
        creds = Credentials.from_authorized_user_file(
            os.path.join(TOKEN_DIR, "token.json"), SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.path.join(TOKEN_DIR, "credentials_wso.json"), SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(os.path.join(TOKEN_DIR, "token.json"), "w") as token:
            token.write(creds.to_json())
    print("✅ Credentials authenticated successfully.")
    return creds


def build_drive(creds):
    """Builds the Google Drive service."""
    try:
        service = build("drive", "v3", credentials=creds)
        return service
    except Exception as e:
        print(f"❌ Error building Google Drive service: {e}")
        return None


def list_files(creds, page_size=1000, query=None) -> list:
    """Lists the first `page_size` files in the user's Drive."""
    service = build_drive(creds)
    if query is None:
        query = "'root' in parents"
    else:
        query = f"'{query}' in parents"
    results = (
        service.files()
        .list(
            pageSize=page_size,
            fields="nextPageToken, files(id, name, size, modifiedTime, mimeType)",
            q=query,
        )
        .execute()
    )
    items = results.get("files", [])
    return items


def list_all_files(
    creds,
    fields="nextPageToken, files(id, name, size, modifiedTime, mimeType)",
    query=None,
    corpora="user",
    drive_id=None,
    include_items_from_all_drives=False,
    supports_all_drives=False,
) -> list:
    """Lists all files in the user's Drive."""
    try:
        items, token = [], None
        drive = build_drive(creds)
        kwargs = dict(
            pageSize=1000,
            fields=fields,
            corpora=corpora,
            includeItemsFromAllDrives=include_items_from_all_drives,
            supportsAllDrives=supports_all_drives,
            q=query,
        )
        if drive_id:
            kwargs["driveId"] = drive_id
            kwargs["corpora"] = "drive"
            kwargs["includeItemsFromAllDrives"] = True
            kwargs["supportsAllDrives"] = True

        while True:
            print(f"ℹ️ Fetching files... Retrieved {len(items)} items so far.")
            if token:
                kwargs["pageToken"] = token
            response = drive.files().list(**kwargs).execute()
            items.extend(response.get("files", []))
            token = response.get("nextPageToken")
            if not token:
                print(f"✅ All files fetched. Total items: {len(items)}")
                break
        return items
    except HttpError as error:
        print(f"An error occurred: {error.resp.status} - {error._get_reason()}")
        return []


def files_dataframes(
    creds, page_size=1000, query=None, all_files=False, folder_id=None, drive_id=None
) -> pd.DataFrame:
    """Returns a DataFrame with the first `page_size` files in the user's Drive."""
    if all_files:
        if folder_id:
            folder_query = f"'{folder_id}' in parents"
            if query:
                folder_query = f"({query}) and '{folder_id}' in parents"
        else:
            folder_query = query
        
        if drive_id:
            items = list_all_files(
                creds,
                fields="nextPageToken, files(id, name, size, modifiedTime, mimeType)",
                query=folder_query,
                corpora="drive",
                drive_id=drive_id,
                include_items_from_all_drives=True,
                supports_all_drives=True,
            )
        else:
            items = list_all_files(
                creds,
                fields="nextPageToken, files(id, name, size, modifiedTime, mimeType)",
                query=folder_query,
                corpora="user",
                drive_id=None,
                include_items_from_all_drives=False,
                supports_all_drives=False,
            )
    else:
        items = list_files(creds, page_size=page_size, query=query)

    if not items:
        return pd.DataFrame(columns=["id", "name"])

    for item in items:
        is_folder = item.get("mimeType") == "application/vnd.google-apps.folder"
        if is_folder:
            subfolder_items = list_files(creds, page_size=page_size, query=item["id"])
            items.extend(subfolder_items)

    data = {
        "id": [item["id"] for item in items],
        "name": [item["name"] for item in items],
        "size_mb": [round(int(item.get("size", 0)) / 1_000_000, 2) for item in items],
        "modifiedTime": [item["modifiedTime"] for item in items],
        "mimeType": [item["mimeType"] for item in items],
    }

    df = pd.DataFrame(data)
    return df


def get_folder_id(service, folder_name):
    """Returns the ID of a folder with the given name."""
    query = (
        f"name='{folder_name}'"
        " and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )
    results = (
        service.files()
        .list(pageSize=1, fields="files(id, name, modifiedTime, mimeType)", q=query)
        .execute()
    )
    items = results.get("files", [])
    if not items:
        return None
    return items[0]["id"]


def get_file_id(service, file_name, folder_id=None):
    """Returns the ID of a file with the given name."""
    query = (
        f"name='{file_name}'"
        " and mimeType!='application/vnd.google-apps.folder'"
        " and trashed=false"
    )

    if folder_id:
        query += f" and '{folder_id}' in parents"

    results = (
        service.files()
        .list(pageSize=1, fields="files(id, name, modifiedTime, mimeType)", q=query)
        .execute()
    )
    items = results.get("files", [])
    if not items:
        return None
    return items[0]["id"]


def list_folders(service, folder_id=None, page_size=10):
    """Lists folders in the user's Drive."""
    if folder_id is None:
        query = "mimeType='application/vnd.google-apps.folder' and trashed=false"
    else:
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"

    results = (
        service.files()
        .list(pageSize=page_size, fields="nextPageToken, files(id, name)", q=query)
        .execute()
    )
    items = results.get("files", [])
    return items


def download_file(creds, file_id, destination):
    """Downloads a file from Google Drive."""
    try:
        service = build("drive", "v3", credentials=creds)
        request = service.files().get_media(fileId=file_id)
        with open(destination, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")
                if done:
                    print("✅ Download Complete!")

    except HttpError as error:
        print(f"An error occurred: {error.resp.status} - {error._get_reason()}")


def delete_file(creds, file_id):
    """Deletes a file from Google Drive."""
    try:
        service = build("drive", "v3", credentials=creds)
        service.files().delete(fileId=file_id).execute()
        print(f"✅ File with ID {file_id} deleted.")
    except HttpError as error:
        print(f"An error occurred: {error.resp.status} - {error._get_reason()}")
    except Exception as e:
        print(f"An error occurred: {e}")


def upload_file(creds, file_path, folder_id=None):
    """Uploads a file to Google Drive with proper error handling."""
    file_name = os.path.basename(file_path)
    file_id = None
    service = build("drive", "v3", credentials=creds)

    # CHECK FILE EXISTENCE
    is_existing = get_file_id(service, file_name)
    if is_existing:
        print(f"⚠️ File already exists in Drive: {file_name}")
        return is_existing

    if not file_path or not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    try:
        file_size = os.path.getsize(file_path)
    except OSError as e:
        print(f"❌ Cannot access file {file_name}: {e}")
        return None

    if file_size == 0:
        print(f"⚠️ Skipping empty file: {file_name}")
        return "EMPTY"

    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    file_metadata = {"name": file_name}
    if folder_id:
        file_metadata["parents"] = [folder_id]

    try:
        print(f"📤 Starting upload: {file_name} ({file_size:,} bytes)")

        if file_size < 5 * 1024 * 1024:
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=False)
            response = (
                service.files()
                .create(body=file_metadata, media_body=media, fields="id,name")
                .execute()
            )

            if response and "id" in response:
                file_id = response.get("id")
                print(f"✅ {file_name} Upload Complete!")
            else:
                print(f"❌ {file_name} Upload failed")
                return None
        else:
            chunk_size = min(1024 * 1024, file_size // 10)
            media = MediaFileUpload(
                file_path, mimetype=mime_type, chunksize=chunk_size, resumable=True
            )
            request = service.files().create(
                body=file_metadata, media_body=media, fields="id,name"
            )

            response = None
            while response is None:
                try:
                    status, response = request.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        print(f"📊 {file_name}: {progress}%", end="\r")
                except Exception as chunk_error:
                    print(f"\n❌ Chunk upload error for {file_name}: {chunk_error}")
                    return None

            if response and "id" in response:
                file_id = response.get("id")
                print(f"\n✅ {file_name} Upload Complete!")
            else:
                print(f"\n❌ {file_name} Upload failed - no response")
                return None

    except HttpError as error:
        print(
            f"❌ HTTP Error uploading {file_name}: {error.resp.status} - {error._get_reason()}"
        )
        return None
    except Exception as e:
        print(f"❌ Error uploading {file_name}: {e}")
        return None

    return file_id

def upload_bulky(
    creds, folder, drive_folder_id=None, max_workers=3, resume_failed=False
):
    """Uploads files in a folder to Google Drive using multithreading."""
    print(f"🟢 Running Bulky Upload!")

    file_names = [
        f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
    ]
    print(f"ℹ️ Found {len(file_names):,} files in the folder: {folder}")

    # CHECK EXISTING FILES
    if drive_folder_id:
        print(f"ℹ️ Checking existing files in folder ID: {drive_folder_id}")
        existing_files_df = files_dataframes(creds, all_files=True, folder_id=drive_folder_id)
        existing_files = set(existing_files_df["name"].tolist())
        duplicated = existing_files_df[existing_files_df['name'].duplicated()]

        if not duplicated.empty:
            print(f"ℹ️ {len(duplicated):,} duplicated files found in the folder.")
            print(duplicated[['name', 'size_mb']])
            print(f"🔴 Deleting duplicated files...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(delete_file, creds, file_id): file_id
                    for file_id in duplicated['id'].tolist()
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(duplicated),
                    desc="Deleting duplicated files",
                ):
                    file_id = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"❌ Exception deleting {file_id}: {e}")
            print(f"🔴 Deletion of duplicated files completed.")

            # RECHECK EXISTING FILES AFTER DELETION
            existing_files_df = files_dataframes(creds, all_files=True, folder_id=drive_folder_id)
            existing_files = set(existing_files_df["name"].tolist())
        else:
            print("No duplicated files found.")

        if not file_names:
            print("No new files to upload.")
            return
        
        # DOWNLOAD EXISTING
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        existing_files_df['lat'] = existing_files_df['name'].apply(generate_coordinates).apply(lambda x: x.split(", ")[0] if x else None)
        existing_files_df['lon'] = existing_files_df['name'].apply(generate_coordinates).apply(lambda x: x.split(", ")[1] if x else None)
        existing_files_df['link_foto'] = existing_files_df['id'].apply(generate_link)
        existing_files_df.columns = existing_files_df.columns.str.strip().str.upper()
        existing_files_df.to_csv(os.path.join(script_dir, f"{timestamp}_GDrive_{drive_folder_id}_{os.path.basename(folder).split('.')[0]}.csv"), index=False)

        file_names = [f for f in file_names if f not in existing_files]
        print(f"ℹ️ {len(existing_files):,} existing files found in the folder.")
        print(f"ℹ️ {len(file_names):,} new files to upload.")

    for filename in file_names:
        file_path = os.path.join(folder, filename)
        try:
            file_size = os.path.getsize(file_path)
            # print(f"ℹ️ File: {filename} | Size: {file_size / 1_000_000:.2f} MB")
        except OSError as e:
            print(f"⚠️ Cannot access {filename}: {e}")

    if resume_failed:
        print("ℹ️ Resume True, resuming from previous failed uploads.")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        failed_files = [
            f
            for f in os.listdir(script_dir)
            if f.startswith("failed_uploads_") and f.endswith(".csv")
        ]
        print(f"Script Dir: {script_dir}")
        if failed_files:
            latest_failed = max(
                failed_files, key=lambda x: os.path.getctime(os.path.join(folder, x))
            )
            print(f"ℹ️ Found latest failed upload file: {latest_failed}")
            failed_df = pd.read_csv(os.path.join(folder, latest_failed))
            failed_list = failed_df["Failed Files"].tolist()
            file_names = [f for f in file_names if f in failed_list]
            print(f"ℹ️ Resuming upload for {len(file_names):,} failed files.")
        else:
            print("ℹ️ No previous failed upload file found, uploading all files.")

    total_files = len(file_names)
    print(f"ℹ️ Found {total_files:,} files to upload.")

    successful_uploads = 0
    failed_uploads = 0
    failed_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for file_name in tqdm(file_names, desc="Submitting uploads"):
            file_path = os.path.join(folder, file_name)
            future = executor.submit(upload_file, creds, file_path, drive_folder_id)
            futures[future] = file_name

        for future in tqdm(
            as_completed(futures), total=len(file_names), desc="Uploading files"
        ):
            file_name = futures[future]
            try:
                result = future.result(timeout=600)
                if result and result != "EMPTY":
                    successful_uploads += 1
                else:
                    failed_uploads += 1
                    failed_list.append(file_name)
            except Exception as e:
                failed_uploads += 1
                failed_list.append(file_name)
                print(f"❌ Exception uploading {file_name}: {e}")

    print(f"\n🎉 Upload Summary:")
    print(f"   ✅ Successful: {successful_uploads}")
    print(f"   ❌ Failed: {failed_uploads}")
    print(f"   📊 Total: {successful_uploads + failed_uploads}")

    failed_df = pd.DataFrame(failed_list, columns=["Failed Files"])
    if not failed_df.empty:
        print(f"\n❌ Failed Uploads:")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_filename = f"failed_uploads_{timestamp}.csv"
        failed_path = os.path.join(script_dir, failed_filename)
        failed_df.to_csv(failed_path, index=False)
        print(f"   📁 Saved failed uploads to: {failed_path}")


# UTILITIES
def generate_link(id):
    link = f"https://drive.google.com/file/d/{id}/view?usp=drivesdk"
    return link

def generate_coordinates(name):
    try:
        parts = name.split("_")
        if len(parts) >= 2:
            lat_part = parts[-2]
            lon_part = parts[-1]
            for ext in [".jpg", ".jpeg", ".png"]:
                lat_part = lat_part.replace(ext, "")
                lon_part = lon_part.replace(ext, "")
            
            lat = float(lat_part)
            lon = float(lon_part)
            return f"{lat}, {lon}"
    except Exception as e:
        return None

if __name__ == "__main__":
    try:
        creds = authenticate_credentials()
        folder_path = FOLDER_PATH
        drive_folder_id = DRIVE_ID
        upload_bulky(
            creds,
            folder_path,
            drive_folder_id=drive_folder_id,
            max_workers=48,
            resume_failed=True,
        )

    except Exception as e:
        print(f"❌ An error occurred: {e}")
