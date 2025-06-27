
task=$1
export PYTHONPATH=$(pwd)



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



python a1_gather_sense.py --json_file ./gather_json/all_train.json --count_file ./resources/all_train_count.countpkl --output_file ./resources/all_train_""$task""_1000.combinepkl  --teacher_ckpt_path $teacher_ckpt_path --k 1000 --num_labels $num_labels

python a1_kmeans.py --input_file  ./resources/all_train_""$task""_1000.combinepkl --output_file ./resources/all_train_""$task"".kmeanspkl --k 20 



