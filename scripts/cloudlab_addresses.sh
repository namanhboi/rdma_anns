SOURCED=0
(return 0 2>/dev/null) && SOURCED=1


ALL_CLOUDLAB_HOSTS=(
    "namanh@er087.utah.cloudlab.us"
    "namanh@er072.utah.cloudlab.us"
    "namanh@er092.utah.cloudlab.us"
    "namanh@er041.utah.cloudlab.us"
    "namanh@er083.utah.cloudlab.us" 
    "namanh@er051.utah.cloudlab.us"
    "namanh@er024.utah.cloudlab.us"
    "namanh@er045.utah.cloudlab.us"
    "namanh@er009.utah.cloudlab.us"
    "namanh@er046.utah.cloudlab.us"
    "namanh@er131.utah.cloudlab.us"
)

if [[ $SOURCED == 0 ]]; then
    echo "Cloudlab hosts are: "
    for CLOUDLAB_HOST in ${ALL_CLOUDLAB_HOSTS[@]}; do
	echo "$CLOUDLAB_HOST"
    done
    
fi

export ALL_CLOUDLAB_HOSTS
