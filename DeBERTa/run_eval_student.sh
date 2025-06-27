
task=$1
export PYTHONPATH=$(pwd)

echo "-------------------"
echo $task
echo "-------------------"



case $task in
  MNLI)
  num_labels=3
  ;;
  STS-B)
  num_labels=1
  ;;
  *)
  num_labels=2
esac


case $task in 
    CoLA)
    parameters="--fp16 True \
    --cls_drop_out 0.1 "
    ;;
    MNLI)
    parameters="--fp16 True \
    --max_seq_len 256 "
    ;;
    MRPC)
    parameters="--fp16 True \
    --max_seq_len 128     \
    --cls_drop_out 0.2  "
    ;;
    QNLI)
    parameters="--fp16 True \
    --max_seq_len 512"
    ;;
    QQP)
    parameters="--fp16 True \
    --max_seq_len 320"
    ;;
    RTE)
    parameters="--fp16 True \
    --max_seq_len 320"
    ;;
    SST-2)
    parameters="--fp16 True \
    --max_seq_len 128 "
    ;;
    STS-B)
    parameters=" --fp16 True \
    --max_seq_len 128 "
    ;;

esac

echo "-------------------"
echo "student new" $task
echo "-------------------"

mkdir -p ./results_allstudent/SKD/${task}/


python a0b_run_studentmodel.py \
--model_config ./experiments/glue/config.json  \
--tag deberta-v3-xsmall \
--do_eval \
--task_name $task \
--data_dir ./glue/$task \
--init_model deberta-v3-xsmall \
--output_dir ./results_allstudent/SKD/${task}  \
--student_ckpt_path ./ckpts/SKD/${task}.ckpt  \
--cluster_path ./sense_dict/${task}.kmeanspkl 

