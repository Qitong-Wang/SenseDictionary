
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


# Modify the checkpoints of fine-tuned DeBERTa model.
case $task in
  CoLA)
  teacher_ckpt_path="./ckpts/teacher-v3-large/CoLA/pytorch.model-001072.bin"
  ;;
  MNLI)
  teacher_ckpt_path="./ckpts/teacher-v3-large/MNLI/pytorch.model-011000.bin"
  ;;
  MRPC)
  teacher_ckpt_path="./ckpts/teacher-v3-large/MRPC/pytorch.model-000460.bin"
  ;;
  QNLI)
  teacher_ckpt_path="./ckpts/teacher-v3-large/QNLI/pytorch.model-003000.bin"
  ;;
  QQP)
  teacher_ckpt_path="./ckpts/teacher-v3-large/QQP/pytorch.model-017000.bin"
  ;;
  RTE)
  teacher_ckpt_path="./ckpts/teacher-v3-large/RTE/pytorch.model-000312.bin"
  ;;
  SST-2)
  teacher_ckpt_path="./ckpts/teacher-v3-large/SST-2/pytorch.model-012000.bin"
  ;;
  STS-B)
  teacher_ckpt_path="./ckpts/teacher-v3-large/STS-B/pytorch.model-000718.bin"
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



mkdir -p ./results_teacher/${task}/
mkdir -p ./output_teacher/



echo "-------------------"
echo "LLM Replacement"  $task $cluster
echo "-------------------"

python a0a_run_teachermodel.py \
--model_config ./experiments/glue/config.json  \
--tag deberta-v3-large \
--do_eval \
--task_name $task \
--data_dir ./glue/$task \
--init_model deberta-v3-large \
--output_dir ./results_teacher/${task}/ \
$parameters \
--teacher_ckpt_path $teacher_ckpt_path \
--cluster_path ./sense_dict/$task.kmeanspkl  
