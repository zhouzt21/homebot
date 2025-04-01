#!/bin/bash
# filepath: extract_all.sh

# 指定目标文件夹
TARGET_DIR="/home/zhouzhiting/Projects/homebot/asset/mani_skill2_ycb/models"
# 创建输出目录
OUTPUT_DIR="$TARGET_DIR/extracted"
mkdir -p "$OUTPUT_DIR"

# 遍历所有 .tar.gz 文件并解压
for file in "$TARGET_DIR"/*.tgz; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .tar.gz)
        echo "Extracting: $filename"
        mkdir -p "$OUTPUT_DIR/$filename"
        tar -xzf "$file" -C "$OUTPUT_DIR/$filename"
    fi
done

# TODO: 注意需要修改，使得文件夹名字不含后缀（如 _berkeley_meshes.tgz）; 同时需要修改路径，避免重新文件夹名