#!/bin/bash

# FTP上传模型脚本
# 服务器: ftpuser@192.168.50.102
# 密码: 19790204
# 上传文件: decision_tree_model.pkl

FTP_HOST="192.168.50.102"
FTP_USER="ftpuser"
FTP_PASS="19790204"
MODEL_FILE="/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/models/decision_tree/decision_tree_model.pkl"
REMOTE_DIR="/upload"  # 默认上传到/upload目录
FILENAME=$(basename "$MODEL_FILE")

# 如果通过参数指定了远程目录，使用参数值
if [ -n "$1" ]; then
    REMOTE_DIR="$1"
fi

# 检查文件是否存在
if [ ! -f "$MODEL_FILE" ]; then
    echo "错误: 文件不存在: $MODEL_FILE"
    exit 1
fi

echo "开始上传文件: $MODEL_FILE"
echo "目标服务器: $FTP_USER@$FTP_HOST"
echo "远程目录: $REMOTE_DIR"
echo "远程文件名: $FILENAME"

# 方法1: 尝试使用curl (更可靠)
if command -v curl &> /dev/null; then
    echo "使用curl上传（PASV模式）..."
    
    REMOTE_PATH="${REMOTE_DIR%/}/$FILENAME"  # 确保路径格式正确
    echo "上传到: $REMOTE_PATH"
    
    OUTPUT=$(curl -T "$MODEL_FILE" "ftp://$FTP_HOST$REMOTE_PATH" --user "$FTP_USER:$FTP_PASS" --ftp-pasv 2>&1)
    CURL_EXIT=$?
    
    echo "$OUTPUT"
    
    # 检查是否成功（退出码为0且没有错误信息）
    if [ $CURL_EXIT -eq 0 ] && ! echo "$OUTPUT" | grep -qi "error\|failed\|553\|550\|530"; then
        echo "✅ 上传成功: $FILENAME 到 $REMOTE_PATH"
        exit 0
    else
        echo "curl上传失败，尝试使用ftp命令..."
    fi
fi

# 方法2: 使用ftp命令 (备用方案)
echo "使用ftp命令上传..."
echo "使用PASV模式..."

# 创建临时文件存储FTP输出
FTP_OUTPUT=$(mktemp)

# 执行FTP上传并捕获输出
{
    echo "user $FTP_USER $FTP_PASS"
    echo "binary"
    echo "passive"
    echo "cd $REMOTE_DIR"
    echo "pwd"
    echo "put $MODEL_FILE $FILENAME"
    echo "quit"
} | ftp -n -v "$FTP_HOST" 2>&1 | tee "$FTP_OUTPUT"

# 检查FTP输出中是否有错误
if grep -q "553\|550\|530\|Could not create\|Permission denied\|failed" "$FTP_OUTPUT"; then
    echo ""
    echo "❌ 上传失败!"
    echo "请检查:"
    echo "  1. FTP服务器是否启用了write_enable=YES"
    echo "  2. 用户 $FTP_USER 是否有写入权限"
    echo "  3. 目录 $REMOTE_DIR 是否存在且有写权限"
    echo "  4. 网络连接是否正常"
    rm -f "$FTP_OUTPUT"
    exit 1
elif grep -q "226 Transfer complete\|226 File receive OK" "$FTP_OUTPUT"; then
    echo ""
    echo "✅ 上传成功: $FILENAME 到 $REMOTE_DIR"
    rm -f "$FTP_OUTPUT"
    exit 0
else
    echo ""
    echo "⚠️ 上传状态不确定，请检查FTP服务器"
    rm -f "$FTP_OUTPUT"
    exit 1
fi
