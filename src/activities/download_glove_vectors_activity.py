import zipfile
import requests
from tqdm import tqdm
from pathlib import Path
from temporalio import activity
from dataclasses import dataclass

@dataclass
class DownloadGloveVectorsIn:
    target_dir: str
    embedding_dim: int = 300

@dataclass
class DownloadGloveVectorsOut:
    glove_file_path: str
    target_dir: str
    success: bool
    message: str

@activity.defn
async def download_glove_vectors_activity(data: DownloadGloveVectorsIn) -> DownloadGloveVectorsOut:
    try:
        target_dir = Path(data.target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = target_dir / "glove.6B.zip"
        glove_file_path = target_dir / f"glove.6B.{data.embedding_dim}d.txt"
        
        if glove_file_path.exists():
            return DownloadGloveVectorsOut(
                glove_file_path=str(glove_file_path),
                target_dir=str(target_dir),
                success=True,
                message="GloVe vectors already exist. Skipping download."
            )
        
        if not zip_path.exists():
            url = "https://nlp.stanford.edu/data/glove.6B.zip"
            
            print(f"Downloading GloVe vectors to {target_dir}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            chunk_size = 8192

            with open(zip_path, "wb") as f, tqdm(
                desc="Downloading GloVe",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    bar.update(len(chunk))
            
            print("Download completed.")
        else:
            print("Zip file already exists. Skipping download.")
        
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        
        print(f"Files extracted to: {target_dir}")
        
        if not glove_file_path.exists():
            raise FileNotFoundError(f"Expected embedding file not found: {glove_file_path}")
        
        return DownloadGloveVectorsOut(
            glove_file_path=str(glove_file_path),
            target_dir=str(target_dir),
            success=True,
            message="GloVe vectors downloaded and extracted successfully."
        )
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error during download: {str(e)}"
        print(f"❌ {error_msg}")
        return DownloadGloveVectorsOut(
            glove_file_path="",
            target_dir=str(data.target_dir),
            success=False,
            message=error_msg
        )
        
    except zipfile.BadZipFile as e:
        error_msg = f"Corrupted zip file: {str(e)}"
        print(f"❌ {error_msg}")
        return DownloadGloveVectorsOut(
            glove_file_path="",
            target_dir=str(data.target_dir),
            success=False,
            message=error_msg
        )
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return DownloadGloveVectorsOut(
            glove_file_path="",
            target_dir=str(data.target_dir),
            success=False,
            message=error_msg
        )
