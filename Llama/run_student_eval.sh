export PYTHONPATH=$(pwd)

task_name=$1



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
# Validate input
if [[ $1 -ge 1 && $1 -le ${#task_names[@]} ]]; then
  task_name=${task_names[$(( $1 - 1 ))]}
  echo "The task name for \$1=$1 is: $task_name"
else
  echo "Invalid input. Please enter a number between 1 and ${#task_names[@]}."
fi





echo "----------------------------------------------------"
echo "Evaluate" $task_name
echo "----------------------------------------------------"


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
--ckpt_path ./ckpt/Llama_SKD.pth \
--run_option replace  


cat $PRINT_PATH



