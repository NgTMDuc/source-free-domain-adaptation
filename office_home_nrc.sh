DATASET="office-home" # imagenet_c domainnet126 officehome
METHOD="nrc"          # shot nrc plue difo ProDe

echo DATASET: $DATASET
echo METHOD: $METHOD


# source_free_adaptation() {
# if [ "$DATASET" == "office-home" ];then
s_list=(0 1 2 3)
t_list=(0 1 2 3)
# t_list=(3)
# fi
for s in ${s_list[*]}; do
    for t in ${t_list[*]}; do
    # t=3
    if [[ $s = $t ]]
        then
        continue
    fi
        CUDA_VISIBLE_DEVICES=2 python image_target_of_oh_vs.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
            SETTING.S "$s" SETTING.T "$t" &
        wait
    done
done 

