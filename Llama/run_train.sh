export PYTHONPATH=$(pwd)







echo "----------------------------------------------------"
echo  "Train all"
echo "----------------------------------------------------"



CKPT_PATH=./ckpt/train_all/


mkdir -p $CKPT_PATH

torchrun --nproc_per_node=3 --nnodes=3  --master_addr=$1 --master_port=$2 --node_rank=$3 a2_train_multi.py --model_name McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
--task_name train_all \
--task_to_instructions_fp test_configs/mteb/task_to_instructions.json \
--sense_func dot \
--run_option train \
--lr 0.000005  \
--student_layers 4  \
--json_path ./text_train/train_all/train.json \
--epoch 5 \
--ckpt_path $CKPT_PATH \
--world_size 9



# if $3 (node rank) is 0, evaluate the model
if [ "$3" -eq 0 ]; then

echo "----------------------------------------------------"
echo "Evaluate"
echo "----------------------------------------------------"


task_names=(
  "ArxivClassification"  #1
  "Banking77Classification"
  "DBpediaClassification" #3
  "EmotionClassification"
  "FinancialPhrasebankClassification" #5
  "FrenkEnClassification"
  "ImdbClassification" #7
  "CanadaTaxCourtOutcomesLegalBenchClassification"
  "NewsClassification" #9
  "PatentClassification" 
  "PoemSentimentClassification" #11
  "ToxicChatClassification"
  "ToxicConversationsClassification" #13
  "TweetSentimentExtractionClassification"
  "TweetTopicSingleClassification" #15
  "YelpReviewFullClassification"
)





for task_name in "${task_names[@]}"; do


OUTPUT_PATH=./result/
mkdir -p $OUTPUT_PATH

PRINT_PATH=$OUTPUT_PATH"McGill-NLP__LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/no_revision_available/""$task_name"".json"
rm -rf $PRINT_PATH

python a2_train_multi.py --model_name McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
--task_name $task_name \
--task_to_instructions_fp test_configs/mteb/task_to_instructions.json \
--output_dir $OUTPUT_PATH \
--sense_func dot \
--student_layers 4  \
--epoch 4 \
--ckpt_path $CKPT_PATH/4.pth \
--run_option replace  


cat $PRINT_PATH

done

