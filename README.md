# Graffiti Detection: Object Detection with TensorFlow

![Sample Exported Image](https://github.com/abundis-rmn2/graffiti_detection_OD_TF/blob/main/exported_images/2335768089991704881_25300513537_exported.jpg)

This project identifies graffiti styles and general objects using TensorFlow and a custom-trained object detection model. It supports two approaches: running inference via Jupyter notebooks or as standalone Python scripts. Below are the details and usage instructions for the project.

## Features

1. **Graffiti Style Detection**
   - Custom label map for graffiti styles:
     - Caracter
     - Tag
     - Bomba
     - Roller
     - Wildstyle
     - 3D
     - Moniker
     - S_Tren

2. **Object Detection**
   - Uses the Open Images V4 dataset to detect objects from the world.

3. **Database and FTP Integration**
   - Fetch images and metadata from an SQL database.
   - Upload processed inference results to an FTP server.

---

## Installation

### Prerequisites
Ensure the following are installed on your system:
- Python 3.7+
- TensorFlow (>=2.0)
- MySQL Connector for Python
- Required libraries from `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

---

## Usage

### Jupyter Notebooks

#### 1. **`Inference_SQL.ipynb`**
- Purpose: Performs inference and integrates with a custom database.
- Steps:
  1. Run all setup cells to install dependencies (TensorFlow, MySQL connector, etc.).
  2. Update the parameter `MUID` in the notebook to a valid value obtained from your database.
  3. Ensure database connectivity is configured in the `config.json` file.

#### 2. **`Inference_TF_faster_rcnn.ipynb`**
- Purpose: Inference using the Open Images V4 dataset.
- Steps:
  1. Run all cells to set up the environment.
  2. Ensure the required models are present in the `inference_graph/saved_model` directory.

### Standalone Python Script

#### **`inference-sql-argparse.py`**

- Purpose: Perform inference via command-line arguments with SQL and FTP integration.
- Steps:
  1. Install requirements and dependencies.
  2. Run the script with the necessary arguments:
     ```bash
     python inference-sql-argparse.py -MUID <Your_MUID>
     ```
  3. Ensure the following files are configured:
     - `config.json`: Contains SQL and FTP credentials.
     - `labelmap.pbtxt`: Custom label map for graffiti styles.

---

## Configuration

### SQL Database
- The script connects to the database to fetch metadata and image URLs.
- Example connection code:
  ```python
  cnx = mysql.connector.connect(
      user=config["SQL"]["username"],
      password=config["SQL"]["password"],
      host=config["SQL"]["hostname"],
      database=config["SQL"]["database"]
  )
  ```
- Ensure you update the `config.json` file with correct database credentials:
  ```json
  {
      "SQL": {
          "username": "your_username",
          "password": "your_password",
          "hostname": "your_hostname",
          "database": "your_database"
      }
  }
  ```

### FTP Server
- After inference, processed images and JSON results are uploaded to an FTP server.
- Example upload function:
  ```python
  def DataUpload(local_dir, target_dir):
      ftp_server = ftplib.FTP(
          config["FTP"]["hostname"],
          config["FTP"]["username"],
          config["FTP"]["password"]
      )
      ftp_server.encoding = "utf-8"
      ftp_server.cwd('/media/exported_images')
      # Create target directory if it doesn't exist
      if directory_exists(target_dir, ftp_server) is False:
          ftp_server.mkd(target_dir)
      ftp_server.cwd(target_dir)
      # Upload files
      for filename in os.listdir(local_dir):
          with open(os.path.join(local_dir, filename), 'rb') as file:
              ftp_server.storbinary(f'STOR {filename}', file)
      ftp_server.quit()
  ```
- Update FTP credentials in the `config.json` file:
  ```json
  {
      "FTP": {
          "hostname": "your_ftp_hostname",
          "username": "your_ftp_username",
          "password": "your_ftp_password"
      }
  }
  ```

---

## Directory Structure

```plaintext
project_root/
├── config.json           # Configuration file for SQL and FTP
├── requirements.txt      # Dependencies
├── labelmap.pbtxt        # Label map for graffiti styles
├── inference_graph/      # TensorFlow saved models
│   └── saved_model/
├── notebooks/
│   ├── Inference_SQL.ipynb
│   └── Inference_TF_faster_rcnn.ipynb
├── scripts/
│   └── inference-sql-argparse.py
└── exported_images/      # Processed inference results
```

---

## Sample Results

- Insert sample images and JSON outputs here to demonstrate the results.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributions

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## Contact

For inquiries or support, please contact the project maintainer.
