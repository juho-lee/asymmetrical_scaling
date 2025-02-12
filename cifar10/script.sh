for i in 1 2 3 4 5;
do
    CUDA_VISIBLE_DEVICES=$3 python run.py \
        --alpha $1 \
        --a $2 \
        --expid run$i \
        --gpu $3 \
        --num_iter 100

done