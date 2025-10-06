import pandas as pd
import hashlib
import json
from datetime import datetime
from temporalio import activity

from src.default_types import AcademicWork, ProcessGoogleDriveFileIn, ProcessGoogleDriveFileOut

@activity.defn
async def process_google_drive_file(data: ProcessGoogleDriveFileIn) -> ProcessGoogleDriveFileOut:
  url = f"https://drive.google.com/uc?id={data.file_id}"
  data = []

  try:
    df = pd.read_csv(url)
  except Exception as e:
    activity.logger.error(f"Erro ao ler {data.file_name}: {e}")
    return []

  for col in ['title', 'abstract', 'label_included', 'keywords']:
    if col not in df.columns:
      activity.logger.warning(f"Aviso: '{col}' ausente em {data.file_name}.")
      df[col] = ""

  db_source = data.file_name.replace('.csv', '').lower().replace('-', '_')

  for _, row in df.iterrows():  
    metadata = {
      "title": row['title'],
      "db_source": db_source,
    }
    work_id = hashlib.sha256(json.dumps(metadata).encode()).hexdigest()
    
    work = AcademicWork(
      id=work_id,
      title=row['title'],
      abstract=row['abstract'],
      keywords=row['keywords'] if 'keywords' in df.columns else "",
      included=row['label_included'],
      db_source=db_source,
      created_at=datetime.now(),
      updated_at=datetime.now(),
    )
    data.append(work)
  
  pd.DataFrame(data).to_csv(f"data/{db_source}.csv", index=False)

  activity.logger.info(f"Processados {len(data)} trabalhos de {data.file_name}")

  return ProcessGoogleDriveFileOut(
    file_name=data.file_name,
  )
