import pandas as pd
from datetime import datetime
from typing import Dict
import hashlib
import json

from utils import setup_project_path
setup_project_path()

from constants import SOURCE_INPUT_FILES
from src.default_types import AcademicWork

data = []

def generate_deterministic_id(metadata: Dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(metadata).encode()).hexdigest()

for file_name, file_id in SOURCE_INPUT_FILES:
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Erro ao ler {file_name}: {e}")
        continue

    for col in ['title', 'abstract', 'label_included', 'keywords']:
        if col not in df.columns:
            print(f"Aviso: '{col}' ausente em {file_name}.")
            df[col] = ""

    for _, row in df.iterrows():
        db_source = file_name.replace('.csv', '').lower().replace('-', '_')
        work = AcademicWork(
            id=generate_deterministic_id({
              "title": row['title'],
              "db_source": db_source,
            }),
            title=row['title'],
            abstract=row['abstract'],
            keywords=row['keywords'] if 'keywords' in df.columns else "",
            included=row['label_included'],
            db_source=db_source,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        data.append(work)

df_final = pd.DataFrame(data)
df_final.to_csv('data/academic_works.csv', index=False)

print(f"✅ {len(df_final)} trabalhos salvos em data/academic_works.csv")
