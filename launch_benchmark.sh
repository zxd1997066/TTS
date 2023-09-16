#!/bin/bash
set -xe

# TTS
function prepare_workload {
    # prepare workload
    workload_dir="${PWD}"
    # set common info
    source oob-common/common.sh
    init_params $@
    fetch_device_info
    set_environment

    pip install -r ${workload_dir}/requirements.txt
    pip uninstall -y numba llvmlite
    conda install -c numba llvmdev -y
    pip install git+https://github.com/numba/llvmlite.git
    pip install -U numba

    pip uninstall -y TTS
    python setup.py clean
    python setup.py install
    pip install libtool
    
    if [ $(espeak-ng --help > /dev/null 2>&1 && echo $? || echo $?) -ne 0 ];then
        wget -q -O 1.51.tar.gz https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.tar.gz
        tar xvf 1.51.tar.gz
        cd espeak-ng-1.51/
        ./autogen.sh
        ./configure --prefix=${WORKSPACE}/_install
        make
        make install
        cd ..
        export PATH="${PATH}:${WORKSPACE}/_install/bin"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${WORKSPACE}/_install/lib/"
        export LD_INCLUDE_PATH="$LD_INCLUDE_PATH:${WORKSPACE}/_install/include/"
    fi
}

function main {
    # prepare workload
    prepare_workload $@

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # pre run
        python benchmark.py --model-path ${CKPT_DIR}/best_model.pth.tar --config-path=${CKPT_DIR}/config.json \
            --dataset-name=ljspeech --metadata-path=${DATASET_DIR}/ljspeech/truncated_metadata.csv \
            --perf-num-warmup 1 --perf-num-iters 2  --batch-size 1 \
            --precision ${precision} --channels_last ${channels_last} || true
        #
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean
            # generate launch script for multiple instance
            if [ "${OOB_USE_LAUNCHER}" == "1" ] && [ "${device}" != "cuda" ];then
                generate_core_launcher
            else
                generate_core
            fi
            # launch
            echo -e "\n\n\n\n Running..."
            cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${OOB_EXEC_HEADER} \
            python benchmark.py --model-path ${CKPT_DIR}/tacotron2/best_model.pth.tar \
                --config-path=${CKPT_DIR}/tacotron2/config.json --dataset-name=ljspeech \
                --metadata-path=${DATASET_DIR}/ljspeech/truncated_metadata.csv \
                --perf-num-warmup ${num_warmup} --perf-num-iters ${num_iter} \
                --batch-size ${batch_size} \
                --precision ${precision} \
                --channels_last ${channels_last} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# run
function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m oob-common.launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#cpu_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            benchmark.py --model-path ${CKPT_DIR}/tacotron2/best_model.pth.tar \
                --config-path=${CKPT_DIR}/tacotron2/config.json --dataset-name=ljspeech \
                --metadata-path=${DATASET_DIR}/ljspeech/truncated_metadata.csv \
                --perf-num-warmup ${num_warmup} --perf-num-iters ${num_iter} \
                --batch-size ${batch_size} \
                --precision ${precision} \
                --channels_last ${channels_last} \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git

# Start
main "$@"
