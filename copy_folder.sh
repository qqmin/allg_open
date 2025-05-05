#!/bin/bash

# 描述: 复制文件夹到新路径 保存
# 使用方法: ./copy_folder.sh /path/to/source/folder /path/to/target/folder
# 使用 chmod +x copy_folder.sh 命令使脚本可执行

# 定义 usage 函数, 用于打印使用说明
usage() {
    echo "用法: $0 <源文件夹路径> <目标文件夹路径>"
    exit 1
}

# 检查输入参数数量
if [ "$#" -ne 2 ]; then
    usage
fi

# 获取输入参数
source_folder=$1
target_folder=$2

# 检查源文件夹是否存在
if [ ! -d "$source_folder" ]; then
    echo "错误: 源文件夹不存在"
    exit 1
fi

# 检查目标文件夹的父目录是否存在
target_parent_folder=$(dirname "$target_folder")
if [ ! -d "$target_parent_folder" ]; then
    read -p "<$target_parent_folder> 目标文件夹的父目录不存在, 是否创建它? (y/n): " create_folder
    if [[ $create_folder == "y" ||$create_folder == "Y" ]]; then
        echo "创建目标文件夹的父目录: $target_parent_folder"
        mkdir -p "$target_parent_folder" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "错误: 无法创建目标文件夹的父目录"
            exit 1
        fi
    else
        echo "取消操作"
        exit 1
    fi
fi

echo "********************************************"
echo "准备复制此文件夹: $source_folder"
echo "目标文件夹父目录: $target_parent_folder"
echo "复制后新文件夹名: $(basename "$target_folder")"
echo "********************************************"

# 复制文件夹
cp -r "$source_folder" "$target_parent_folder"

# 检查复制是否成功
if [ $? -eq 0 ]; then
    echo "复制成功: $source_folder"
else
    echo "复制失败: $source_folder"
    exit 1
fi

# 定义一个函数来清理复制的文件夹
cleanup() {
    echo "清理复制的文件夹: $target_parent_folder/$(basename "$source_folder")"
    rm -r "$target_parent_folder/$(basename "$source_folder")"
    if [ $? -eq 0 ]; then
        echo "已删除复制的文件夹: $target_parent_folder/$(basename "$source_folder")"
    else
        echo "删除复制的文件夹失败: $target_parent_folder/$(basename "$source_folder")"
    fi
}

if false; then
    # 设置陷阱来处理中断信号(如 Ctrl+C)
    # 如果在循环中中断脚本, 陷阱会被触发, cleanup 函数会被调用, 从而删除复制的文件夹
    # 设置陷阱来处理中断信号
    trap 'cleanup' INT TERM

    # 重命名复制的文件夹
    target_subfolder=$(basename "$target_folder")
    while [ -d "$target_folder" ]; do
        # echo "警告: 目标文件夹已存在, 请输入一个新的文件夹名称:"
        # read -r new_folder_name
        if [ -z "$new_folder_name" ]; then
            echo "错误: 文件夹名称不能为空"
            continue
        fi
        target_folder="$target_parent_folder/$new_folder_name"
        if [ ! -d "$target_folder" ]; then
            break
        else
            echo "警告: {$target_folder} 文件夹名称已存在, 请输入一个不同的名称"
        fi
    done

    # 移除陷阱
    trap - INT TERM
fi

# 检查目标文件夹是否存在, 如果存在则提示输入新的文件夹名称
if [ -d "$target_folder" ]; then
    echo "-----------------------------------------------------------------------"
    echo "警告: $target_folder 目标文件夹已存在, 请输入一个新的文件夹名称"
    echo "-----------------------------------------------------------------------"
    # read -r new_folder_name
    # target_folder="$target_parent_folder/$new_folder_name"
    cleanup
    exit 1
else
    # 重命名文件夹
    mv "$target_parent_folder/$(basename "$source_folder")" "$target_folder"

    # 检查重命名是否成功
    if [ $? -eq 0 ]; then
        echo "重命名成功: $target_folder"
    else
        echo "重命名失败: $target_folder"
        cleanup
        exit 1
    fi
fi

echo "********************************************"
# 退出脚本
exit 0
