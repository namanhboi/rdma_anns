SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/cloudlab_addresses.sh

for CLOUDLAB_HOST in ${ALL_CLOULAB_HOSTS[@]}; do
    ssh $CLOUDLAB_HOST
done

    
