import shutil
import requests
import zipfile
import io
from pathlib import Path

# Set the repo details
repo_owner = 'gvieralopez'
repo_name = 'craters-data'

url = 'https://github.com/gvieralopez/craters-data/archive/refs/heads/main.zip'
headers = {'Accept': 'application/vnd.github.v3+json'}

# Get the script file path
script_path = Path(__file__).resolve().parent

# Send the API request and get the zip file contents
response = requests.get(url, headers=headers)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Create a temporary directory for extracting the contents of the zip file
temp_dir = script_path / 'temp'
temp_dir.mkdir(exist_ok=True)

# Extract the contents of the zip file to the temporary directory
zip_file.extractall(temp_dir)

# Move the 'data' subfolder from the temporary directory to the current directory
data_dir = temp_dir / 'craters-data-main/data'
shutil.move(str(data_dir), str(script_path))

# Remove the temporary directory and its contents
shutil.rmtree(temp_dir)

# Close the zip file
zip_file.close()