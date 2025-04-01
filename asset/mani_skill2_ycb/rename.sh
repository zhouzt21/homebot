#!/bin/bash

# 指定路径
TARGET_DIR="/home/zhouzhiting/Projects/homebot/asset/mani_skill2_ycb/models"

# 遍历目标路径下的所有文件夹
for dir in "$TARGET_DIR"/*/; do
    # 检查文件夹名称是否包含 _berkeley_meshes 后缀
    if [[ "$dir" == *_berkeley_meshes/ ]]; then
        # 去掉 _berkeley_meshes 后缀
        new_dir="${dir%_berkeley_meshes/}"
        # 重命名文件夹
        mv "$dir" "$new_dir"
    fi
done