#!/usr/bin/env bash
# ШЛИФ-Скан: установка on-prem (air-gapped friendly).
# Вариант А (есть интернет): docker compose up --build
# Вариант Б (без интернета): на машине с интернетом:
#     docker build -t shlifscan . && docker save shlifscan | gzip > shlifscan.tar.gz
#   на целевой машине:
#     ./install.sh shlifscan.tar.gz
set -euo pipefail

if [[ $# -ge 1 && -f "$1" ]]; then
  echo "Загрузка образа из $1 ..."
  gunzip -c "$1" | docker load
fi

mkdir -p runs models
if [[ ! -f models/talc_unet.pt ]]; then
  echo "ВНИМАНИЕ: models/talc_unet.pt не найден — детекция талька будет работать"
  echo "по классической эвристике. Скопируйте веса из архива поставки в models/."
fi

docker compose up -d
echo "ШЛИФ-Скан доступен на http://localhost:8000"
