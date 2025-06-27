task=$1
export PYTHONPATH=$(pwd)


echo "-------------------"
echo $task
echo "-------------------"


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

case $task in
  CoLA)
  teacher_ckpt_path="./ckpts/SKD/teacher-v3-large/CoLA/pytorch.model-001072.bin"
  ;;
  MNLI)
  teacher_ckpt_path="./ckpts/SKD/teacher-v3-large/MNLI/pytorch.model-011000.bin"
  ;;
  MRPC)
  teacher_ckpt_path="./ckpts/SKD/teacher-v3-large/MRPC/pytorch.model-000460.bin"
  ;;
  QNLI)
  teacher_ckpt_path="./ckpts/SKD/teacher-v3-large/QNLI/pytorch.model-003000.bin"
  ;;
  QQP)
  teacher_ckpt_path="./ckpts/SKD/teacher-v3-large/QQP/pytorch.model-017000.bin"
  ;;
  RTE)
  teacher_ckpt_path="./ckpts/SKD/teacher-v3-large/RTE/pytorch.model-000312.bin"
  ;;
  SST-2)
  teacher_ckpt_path="./ckpts/SKD/teacher-v3-large/SST-2/pytorch.model-012000.bin"
  ;;
  STS-B)
  teacher_ckpt_path="./ckpts/SKD/teacher-v3-large/STS-B/pytorch.model-000718.bin"
  ;;
esac


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


sense_dict_path=./sense_dict/${task}.kmeanspkl 

case $task in 
    CoLA)
    max_epoch=40
    lr=0.001
    scheduler=True
    ;;
    MNLI)
    max_epoch=20
    lr=0.001
    scheduler=True
    ;;
    MRPC)
    max_epoch=20
    lr=0.002
    scheduler=True
    ;;
    QNLI)
    max_epoch=20
    lr=0.001
    scheduler=True
    ;;
    QQP)
    max_epoch=20
    lr=0.0005
    scheduler=True
    ;;
    RTE)
    max_epoch=40 
    lr=0.002
    scheduler=False
    ;;
    SST-2)
    lr=0.0005
    max_epoch=20
    scheduler=False
    ;;
    STS-B)
    lr=0.002
    max_epoch=20
    scheduler=True
    ;;

esac


echo "-------------------"
echo "Train"  $task  
echo "-------------------"
mkdir -p ./ckpts/SKD/$task/
torchrun   --nproc_per_node=6 --nnodes=1       a3_traincluster_evalmulti.py --json_path ./resources/glue_json/${task}_train.json --cluster_path $sense_dict_path --teacher_ckpt_path  $teacher_ckpt_path --ckpt_path "./ckpts/SKD/$task/$task""_.ckpt" --num_labels $num_labels --lr $lr --epoch $max_epoch --warmup 0.1 --scheduler $scheduler 



epoch=$((max_epoch - 1)) 


echo "-------------------"
echo "Evaluation"  $task  $epoch
echo "-------------------"



python a0b_run_studentmodel.py \
--model_config ./experiments/glue/config.json  \
--tag deberta-v3-xsmall \
--do_eval \
--task_name $task \
--data_dir ./glue/$task \
--init_model deberta-v3-xsmall \
--output_dir ./results/$task/$epoch   \
$parameters \
--student_ckpt_path "./ckpts/SKD/$task/$task""_""$epoch"".ckpt" \
--cluster_path $sense_dict_path
