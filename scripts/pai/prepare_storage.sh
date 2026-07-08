#!/bin/bash
set -e

NETDISK="/netdisk/liumingxuan"
NETCACHE="/netcache/liumingxuan"

echo "=== Preparing PAI shared storage ==="

mkdir -p "${NETDISK}/models"
mkdir -p "${NETDISK}/data"
mkdir -p "${NETDISK}/saves"
mkdir -p "${NETCACHE}/huggingface"

echo ""
echo "=== Syncing Qwen2.5-7B-Instruct model ==="
if [ -d "$HOME/model/Qwen_2.5-7B-Instruct" ]; then
    rsync -av --progress "$HOME/model/Qwen_2.5-7B-Instruct/" "${NETDISK}/models/Qwen2.5-7B-Instruct/"
    echo "Model synced to: ${NETDISK}/models/Qwen2.5-7B-Instruct"
else
    echo "WARNING: ~/model/Qwen_2.5-7B-Instruct not found"
    echo "You can download it with:"
    echo "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ${NETDISK}/models/Qwen2.5-7B-Instruct"
fi

echo ""
echo "=== Syncing edit data ==="
if [ -d "$HOME/open-unlearning/data" ]; then
    rsync -av --progress "$HOME/open-unlearning/data/" "${NETDISK}/data/"
    echo "Data synced to: ${NETDISK}/data"
fi

echo ""
echo "=== Storage layout ==="
echo "${NETDISK}/"
ls -la "${NETDISK}/"
echo ""
echo "${NETCACHE}/"
ls -la "${NETCACHE}/"
echo ""
echo "=== Done ==="
echo ""
echo "In PAI jobs, these paths are accessible as:"
echo "  /mnt/confignfs/userdata/liumingxuan/  (netdisk)"
echo "  /mnt/confignfs/usercache/liumingxuan/ (netcache)"
