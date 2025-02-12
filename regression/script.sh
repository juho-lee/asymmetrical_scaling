for i in {1..5};
do
    for data in "plant" "naval";
    do
        python run.py \
            --data $data \
            --alpha $1 \
            --a $2 \
            --expid run$i \
            --gpu $3

        python run.py \
            --data $data \
            --alpha $1 \
            --a $2 \
            --expid run$i \
            --gpu $3 \
            --mode prune

        python run.py \
            --data $data \
            --alpha $1 \
            --a $2 \
            --expid run$i \
            --gpu $3 \
            --mode transfer
    done
done
