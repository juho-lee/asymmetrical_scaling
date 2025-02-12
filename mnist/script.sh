for i in {1..5};
do
    python run.py \
        --alpha $1 \
        --a $2 \
        --expid run$i \
        --gpu $3

    python run.py \
        --alpha $1 \
        --a $2 \
        --expid run$i \
        --gpu $3 \
        --mode transfer

    python run.py \
        --alpha $1 \
        --a $2 \
        --expid run$i \
        --gpu $3 \
        --mode prune
done
