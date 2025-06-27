task=$1
export PYTHONPATH=$(pwd)

echo "-------------------"
echo $task
echo "-------------------"


case $task in 
    CoLA)
    parameters="--num_train_epochs 10 \
    --fp16 True \
    --warmup 50 \
    --learning_rate 5.5e-6 \
    --train_batch_size 32 \
    --cls_drop_out 0.1 "
    ;;
    MNLI)
    parameters="--num_train_epochs 2 \
    --fp16 True \
    --warmup 500 \
    --learning_rate 7e-6 \
    --train_batch_size 64 \
    --cls_drop_out 0.3 \
    --max_seq_len 256 \
    --eval_batch_size 256 \
    --dump_interval 1000 "
    ;;
    MRPC)
    parameters="--num_train_epochs 6 \
    --fp16 True \
    --warmup 50 \
    --learning_rate 6e-6  \
    --train_batch_size 32 \
    --max_seq_len 128     \
    --cls_drop_out 0.2  "
    ;;
    QNLI)
    parameters="--num_train_epochs 6 \
    --fp16 True \
    --warmup 200 \
    --learning_rate 7e-6  \
    --train_batch_size 64 \
    --max_seq_len 512     \
    --cls_drop_out 0.2 "
    ;;
    QQP)
    parameters="--num_train_epochs 6 \
    --fp16 True \
    --warmup 1000 \
    --max_seq_len 320 \
    --learning_rate 1e-5  \
    --train_batch_size 64 \
    --cls_drop_out 0.2 "
    ;;
    RTE)
    parameters="--num_train_epochs 6 \
    --fp16 True \
    --warmup 50 \
    --learning_rate 1e-5  \
    --train_batch_size 32 \
    --max_seq_len 320     \
    --cls_drop_out 0.3 "
    ;;
    SST-2)
    parameters="--num_train_epochs 8 \
    --fp16 True \
    --warmup 500 \
    --learning_rate 6e-6  \
    --train_batch_size 32 \
    --cls_drop_out 0.1  \
    --max_seq_len 128 "
    ;;
    STS-B)
    parameters="--num_train_epochs 4 \
    --fp16 True \
    --warmup 50 \
    --learning_rate 7e-6  \
    --train_batch_size 32 \
    --cls_drop_out 0.1 \
    --max_seq_len 128 "
    ;;

esac
# Train the teacher model first
python a0a_run_teachermodel.py \
--model_config ./experiments/glue/config.json  \
--tag deberta-v3-large \
--do_train \
--task_name $task \
--data_dir ./glue/$task \
--init_model deberta-v3-large \
--output_dir ./ckpts/teacher-v3-large/$task   \
$parameters 

