import zipfile
import requests
from tqdm import tqdm
from pathlib import Path

url = "https://nlp.stanford.edu/data/glove.6B.zip"

target_dir = Path("./data/word_vectors/glove")
zip_path = target_dir / "glove.6B.zip"

target_dir.mkdir(parents=True, exist_ok=True)

if not zip_path.exists():
    print("Baixando o arquivo...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192

    with open(zip_path, "wb") as f, tqdm(
        desc="Download",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    print("Download concluído.")
else:
    print("Arquivo já existe. Pulando download.")

print("Descompactando...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(target_dir)
print("Arquivos extraídos para:", target_dir)