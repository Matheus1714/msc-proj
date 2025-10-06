import pandas as pd
import hashlib
import json
from datetime import datetime
from temporalio import activity

from src.default_types import AcademicWork, ProcessGoogleDriveFileIn, ProcessGoogleDriveFileOut

@activity.defn
async def process_files_activity(params: ProcessGoogleDriveFileIn) -> ProcessGoogleDriveFileOut | None:
  # Adicionar debug para ver o que está chegando
  activity.logger.info(f"Tipo de params.file_name: {type(params.file_name)}")
  activity.logger.info(f"Valor de params.file_name: {params.file_name}")
  
  # Verificação de tipo para resolver o problema
  if isinstance(params.file_name, list):
    file_name = params.file_name[0] if params.file_name else ""
    activity.logger.warning(f"file_name era uma lista, convertido para: {file_name}")
  else:
    file_name = params.file_name
    
  url = f"https://drive.google.com/uc?id={params.file_id}"
  works = []

  try:
    df = pd.read_csv(url)
  except Exception as e:
    activity.logger.error(f"Erro ao ler {file_name}: {e}")
    return None

  db_source = (
    file_name
    .replace('.csv', '')
    .lower()
    .replace('-', '_')
  )

  output_path = f"data/{db_source}.csv"

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
    works.append(work)
  
  pd.DataFrame(works).to_csv(output_path, index=False)

  activity.logger.info(f"Processados {len(works)} trabalhos de {file_name}")

  return ProcessGoogleDriveFileOut(
    file_path=output_path,
  )
