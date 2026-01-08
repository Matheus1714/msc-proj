import pandas as pd
import hashlib
import json
from datetime import datetime
from temporalio import activity
from dataclasses import dataclass
from typing import (
  TypedDict,
  List,
  NewType,
)

Id = NewType("Id", str)

class AcademicWork(TypedDict):
  id: str
  title: str
  abstract: str
  keywords: List[str]
  included: bool
  db_source: str
  created_at: datetime
  updated_at: datetime

@dataclass
class ProcessFileIn:
  file_name: str
  file_path: str

@dataclass
class ProcessFileOut:
  file_path: str

@activity.defn
async def process_files_activity(params: ProcessFileIn) -> ProcessFileOut | None:
  activity.logger.info(f"Tipo de params.file_name: {type(params.file_name)}")
  activity.logger.info(f"Valor de params.file_name: {params.file_name}")

  if isinstance(params.file_name, list):
    file_name = params.file_name[0] if params.file_name else ""
    activity.logger.warning(f"file_name era uma lista, convertido para: {file_name}")
  else:
    file_name = params.file_name
    
  works = []

  try:
    df = pd.read_csv(params.file_path)
  except Exception as e:
    activity.logger.error(f"Erro ao ler {file_name}: {e}")
    return None
  
  df = df.dropna(subset = ["title", "abstract", "label_included"])
  # df = df.dropna(subset = ["abstract", "label_included"])

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

  return ProcessFileOut(
    file_path=output_path,
  )
