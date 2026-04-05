SOURCED=0
(return 0 2>/dev/null) && SOURCED=1


ALL_CLOUDLAB_HOSTS=(
    "namanh@hp199.utah.cloudlab.us"
    "namanh@hp177.utah.cloudlab.us"
    "namanh@hp171.utah.cloudlab.us"
    "namanh@hp163.utah.cloudlab.us"
    "namanh@hp196.utah.cloudlab.us"
    "namanh@hp188.utah.cloudlab.us"
)

if [[ $SOURCED == 0 ]]; then
    echo "Cloudlab hosts are: "
    for CLOUDLAB_HOST in ${ALL_CLOUDLAB_HOSTS[@]}; do
	echo "$CLOUDLAB_HOST"
    done
    
fi

export ALL_CLOUDLAB_HOSTS
