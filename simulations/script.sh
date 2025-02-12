for i in {1..5};
do
    python run.py \
        --alpha $1 \
        --a $2 \
        --expid run$i \
        --gpu $3
done
