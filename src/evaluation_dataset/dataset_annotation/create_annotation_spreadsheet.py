import json
from tqdm import tqdm
from pathlib import Path

import pydrive
import pydrive.auth
from pydrive.drive import GoogleDrive
import gspread
from gspread_formatting import *

from src.path import test_intermediate_dir


image_list_jsonl_files = {
    "geometry_diagram": [
        ("MathVista", test_intermediate_dir / "data/geometry_diagram/MathVista.jsonl"),
        ("MathVerse", test_intermediate_dir / "data/geometry_diagram/MathVerse.jsonl"),
    ],
    "chemistry": [
        ("MMMU_Chemistry", test_intermediate_dir / "data/chemistry/MMMU_Chemistry.jsonl"),
        ("MMMU_Chemistry_single", test_intermediate_dir / "data/chemistry/MMMU_Chemistry_single.jsonl")
    ],
    "charts": [
        ("ChartQA", test_intermediate_dir / "data/charts/ChartQA.jsonl"),
        ("CharXiv", test_intermediate_dir / "data/charts/CharXiv.jsonl")
    ],
}


def get_folder_id_by_name(drive, folder_name, parent_folder_id=None):
    query = f"title='{folder_name}'"
    if parent_folder_id is not None:
        query += f" and '{parent_folder_id}' in parents"
    
    file_list = drive.ListFile({'q': query}).GetList()
    
    # clean up
    # remove shortcuts
    file_list = [file for file in file_list if file['mimeType'] == 'application/vnd.google-apps.folder']
    
    if parent_folder_id is not None:
        # direct parent should be the parent_folder_id
        file_list = [file for file in file_list if parent_folder_id == file['parents'][-1]['id']]
    
    if len(file_list) == 0:
        return None
    elif len(file_list) == 1:
        return file_list[0]['id']
    else:
        for file in file_list:
            print(file)
            print()
        
        raise ValueError(f"Multiple folders with the same name {folder_name} exist")


def create_and_upload_drive_folder(drive, folder_name: str, parent_folder_id=None):
    folder_id = get_folder_id_by_name(drive, folder_name, parent_folder_id)
    if folder_id is not None:
        print(f"{folder_name} already exists")
        return folder_id
    
    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_folder_id is not None:
        folder_metadata['parents'] = [{'id': parent_folder_id}]
    
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    return folder['id']


def recursively_create_and_upload_drive_folder(drive, folder_name: str):
    folder_names_list = folder_name.split("/")
    folder_id = None
    for folder_name in folder_names_list:
        folder_id = create_and_upload_drive_folder(drive, folder_name, parent_folder_id=folder_id)
    
    return folder_id


def create_spreadsheet_shortcut(shortcut_title: str, spreadsheet_id: str, target_folder_id: str):
    shortcut_metadata = {
        'title': shortcut_title,
        'mimeType': 'application/vnd.google-apps.shortcut',
        'parents': [{'id': target_folder_id}],
        'shortcutDetails': {
            'targetId': spreadsheet_id,
            'targetMimeType': 'application/vnd.google-apps.spreadsheet'
        }
    }
    
    shortcut = drive.CreateFile(shortcut_metadata)
    shortcut.Upload()


def get_gdrive_url_for_figure_or_upload_figure(drive: GoogleDrive, image_path: str, local_image_dir: Path, drive_dir_name: str):
    # create parent directories of image_path recursively
    parent_dir = Path(image_path).parent
    parent_dir_id = recursively_create_and_upload_drive_folder(drive, f"{drive_dir_name}/{parent_dir}")
    
    # upload image to drive
    # check if the image exists in drive
    image_name = Path(image_path).name
    file_id = get_folder_id_by_name(drive, image_name, parent_folder_id=parent_dir_id)
    if file_id is None:
        image = drive.CreateFile({"title": image_name, 'parents': [{'id': parent_dir_id}]})
        image['sharingUser'] = {'role': 'reader', 'type': 'anyone'}
        image.SetContentFile(f"{local_image_dir}/{image_path}")
        image.Upload()
        file_id = image['id']
    else:
        print(f"{image_name} already exists")
    
    return f"https://drive.google.com/uc?id={file_id}"


if __name__ == "__main__":
    gauth = pydrive.auth.GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    
    # create drive directory
    drive_dir_name = "VisOnlyQA_annotation"
    drive_dir_id = create_and_upload_drive_folder(drive, drive_dir_name)
    
    # spreadsheet
    gc = gspread.service_account(filename='credentials/google_spreadsheet_credential.json')

    with open("credentials/google_account_email.txt", "r") as f:
        email = f.read().strip()

    # 1. Create a directory in a google drive
    # 2. Upload images to the directory
    # 3. Create a spreadsheet
    for image_category in image_list_jsonl_files.keys():
        print(f"Creating a spreadsheet for {image_category} images")
        
        spreadsheet_name = f"VisOnlyQA_{image_category}_annotation"
        
        # check if the spreadsheet exists
        try:
            sh = gc.open(spreadsheet_name)
            print(f"{spreadsheet_name} already exists")
        except:
            sh = gc.create(spreadsheet_name)
            sh.share(email, perm_type='user', role='writer')

            # create a shortcut in the drive directory
            create_spreadsheet_shortcut(sh.title, sh.id, drive_dir_id)
        
        for data_name, jsonl_file in image_list_jsonl_files[image_category]:
            print(f"Adding {data_name} data to the spreadsheet")
            
            with open(jsonl_file, "r") as f:
                image_path_list = [json.loads(line) for line in f]
            image_path_list = image_path_list[:1000]  # limit the number of images to 1000
            
            try:
                sh.add_worksheet(title=data_name, rows=len(image_path_list), cols=5)
                worksheet = sh.worksheet(data_name)
                set_row_height(worksheet, f'1:{len(image_path_list)}', 200)
                set_column_width(worksheet, f'1:{len(image_path_list)}', 200)
            except Exception as e:
                print(e)
                print()
                print(f"Worksheet {data_name} already exists. Continue.")
                continue
            
            for row in tqdm(image_path_list):
                image_path = row["image"]
                
                # upload image to drive
                gdrive_url = get_gdrive_url_for_figure_or_upload_figure(drive, image_path, local_image_dir=test_intermediate_dir, drive_dir_name=drive_dir_name)
                
                # Add a row of "image_path" and "image" to the spreadsheet
                worksheet.append_row([image_path, gdrive_url])
