DATASET="domainnet126" # imagenet_c domainnet126 officehome
METHOD="difo"          # shot nrc plue difo

echo DATASET: $DATASET
echo METHOD: $METHOD


# source_free_adaptation() {
# if [ "$DATASET" == "office-home" ];then
s_list=(0 1 2 3)
t_list=(0 1 2 3)
# fi
for s in ${s_list[*]}; do
    for t in ${t_list[*]}; do
    if [[ $s = $t ]]
        then
        continue
    fi
        python image_target_in_126.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
            SETTING.S "$s" SETTING.T "$t" &
        wait
    done
done 

