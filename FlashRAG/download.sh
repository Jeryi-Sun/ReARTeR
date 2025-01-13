SOURCE_DIR="/home/sunzhongxiang/rag_reasoning/"
# 本地目标文件夹路径
DEST_DIR="/home/sunzhongxiang/RAG_reasoning/"

# 服务器信息
HOST="172.28.205.233"
USER="sunzhongxiang"

# 使用rsync进行增量下载
devctl rsync  $HOST:$SOURCE_DIR $DEST_DIR -p "cQbaX3ymuEOr" -- -av --stats --progress

echo "Download complete!"

