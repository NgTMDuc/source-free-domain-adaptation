DATASET="officehome" # imagenet_c domainnet126 officehome
METHOD="plue"          # shot nrc plue difo ProDe
GPU=2
echo DATASET: $DATASET
echo METHOD: $METHOD


# source_free_adaptation() {
# if [ "$DATASET" == "office-home" ];then
s_list=(0 1 2)
t_list=(0 1 2)
# fi
for s in ${s_list[*]}; do
    for t in ${t_list[*]}; do
    if [[ $s = $t ]]
        then
        continue
    fi
        CUDA_VISIBLE_DEVICES=2 python image_target_of_oh_vs.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
            SETTING.S "$s" SETTING.T "$t" &
        wait
    done
done 

