#!/bin/bash

BASE_DIR="/Users/matheusmota/src/github/msc/systematic-review-datasets"

FILES=(
  "UrinaryIncontinence.csv"
  "Hall_2012.csv"
  "ProtonPumpInhibitors.csv"
  "van_de_Schoot_2017.csv"
  "SkeletalMuscleRelaxants.csv"
  "Kwok_2020.csv"
  "Nagtegaal_2019.csv"
  "Wolters_2018.csv"
  "Kitchenham_2010.csv"
  "Radjenovic_2013.csv"
  "Triptans.csv"
  "Wahono_2015.csv"
  "Statins.csv"
  "van_Dis_2020.csv"
  "ADHD.csv"
  "ACEInhibitors.csv"
  "Opiods.csv"
  "AtypicalAntipsychotics.csv"
  "CalciumChannelBlockers.csv"
  "BetaBlockers.csv"
  "NSAIDS.csv"
  "Bannach-Brown_2019.csv"
  "Appenzeller-Herzog_2020.csv"
  "OralHypoglycemics.csv"
  "Antihistamines.csv"
  "Bos_2018.csv"
  "Estrogens.csv"
)

echo "SOURCE_INPUT_FILES = ["

for file in "${FILES[@]}"; do
  # Busca o caminho completo dentro da pasta
  found_path=$(find "$BASE_DIR" -type f -name "$file" 2>/dev/null | head -n 1)
  
  if [ -n "$found_path" ]; then
    echo "  (\"$file\", \"$found_path\"),"
  else
    echo "  (\"$file\", \"# NÃO ENCONTRADO\"),"
  fi
done

echo "]"
