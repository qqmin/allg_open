#!/bin/bash

# 描述: 清空指定文件夹中的所有文件和子文件夹
# 使用方法: ./clear_folder.sh /path/to/folder
# 使用 chmod +x clear_folder.sh 命令使脚本可执行

# 检查是否提供了文件夹路径
if [ -z "$1" ]; then
    echo "请提供要清空的文件夹路径"
    exit 1
fi

# 检查文件夹是否存在
if [ ! -d "$1" ]; then
    echo "提供的路径不是有效的文件夹"
    exit 1
fi

# 获取文件夹路径
FOLDER_PATH="$1"

# 进入文件夹
cd "$FOLDER_PATH" || {
    echo "无法进入文件夹 $FOLDER_PATH"
    exit 1
}

# 显示警告信息并请求用户确认
read -p "警告: 这将删除文件夹 $FOLDER_PATH 中的所有内容, 确定要继续吗? (y/n): " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 1
fi

# 列出所有文件和文件夹并删除
for ITEM in *; do
    # 检查文件名中是否包含特殊符号(例如连字符)
    if [[ "$ITEM" =~ [-] ]]; then
        echo "文件名 $ITEM 包含 - 符号"
        rm -rf -- "$ITEM"
        echo "文件 $ITEM 已删除"
    else
        if [ -d "$ITEM" ]; then
            # 删除子文件夹
            rm -rf "$ITEM"
            echo "子文件夹 $ITEM 已删除"
        elif [ -f "$ITEM" ]; then
            # 删除文件
            rm -f "$ITEM"
            echo "文件 $ITEM 已删除"
        fi
    fi
done

echo "文件夹 $FOLDER_PATH 已清空"

# 退出脚本
exit 0