CLOUDLAB_HOSTS=(
    "namanh@er039.utah.cloudlab.us"
    "namanh@er082.utah.cloudlab.us"
    "namanh@er076.utah.cloudlab.us"
    "namanh@er050.utah.cloudlab.us"
    "namanh@er104.utah.cloudlab.us"
    "namanh@er124.utah.cloudlab.us"
    "namanh@er001.utah.cloudlab.us"
    "namanh@er040.utah.cloudlab.us"
    "namanh@er032.utah.cloudlab.us"
    "namanh@er126.utah.cloudlab.us"                
)

for CLOUDLAB_HOST in "${CLOUDLAB_HOSTS[@]}"; do
    scp -r ~/.ssh "$CLOUDLAB_HOST:~/"

done
