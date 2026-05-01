SOURCED=0
(return 0 2>/dev/null) && SOURCED=1


ALL_CLOUDLAB_HOSTS=(
    "namanh@amd228.utah.cloudlab.us"
    "namanh@amd222.utah.cloudlab.us"
    "namanh@amd234.utah.cloudlab.us"
    "namanh@amd217.utah.cloudlab.us"
    "namanh@amd230.utah.cloudlab.us"
    "namanh@amd227.utah.cloudlab.us"
    "namanh@amd239.utah.cloudlab.us"
    "namanh@amd236.utah.cloudlab.us"
    "namanh@amd241.utah.cloudlab.us"
    "namanh@amd207.utah.cloudlab.us"
    "namanh@amd213.utah.cloudlab.us"
)

if [[ $SOURCED == 0 ]]; then
    echo "Cloudlab hosts are: "
    for CLOUDLAB_HOST in ${ALL_CLOUDLAB_HOSTS[@]}; do
	echo "$CLOUDLAB_HOST"
    done

fi

export ALL_CLOUDLAB_HOSTS
