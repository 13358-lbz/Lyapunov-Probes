# run P(IK) version of self-confidence evaluation on popQA dataset
cd /data/bozhiluan/factual-confidence-of-llms-main
# models=("mistralai/Mixtral-8x7B-Instruct-v0.1" "tiiuae/falcon-40b" "tiiuae/falcon-40b-instruct")
# models=("mistralai/Mixtral-8x7B-v0.1")


models=("tiiuae/falcon-7b-instruct")
#iterate through models
for model_name_or_path in "${models[@]}"; do
  echo $model_name_or_path
  # for the biggest models, single rank
  python ./src/self_knowledge/main.py \
   --model-name=$model_name_or_path \
   --dataset-path=data/clean_popQA.csv \
   --hidden-scorer-path=models/lama_hidden_scorer/tiiuae/falcon-7b-instruct/lama3_hscorer_24_test.pt \
   --save-path=./results_pik2/tiiuae/falcon-7b-instruct/ \
   --batch-size=32 \
   --log-path=./logs/pt/falcon-7b-instruct/
  wait
done



# models=("tiiuae/falcon-7b-instruct")
# #iterate through models
# for model_name in "${models[@]}"; do
#   echo $model_name
#   for i in 24 26 28 30 31 32 -1; do
#     echo $i
#     # for the biggest models, single rank
#     python ./src/self_knowledge/main.py \
#     --model-name=$model_name \
#     --dataset-path=data/clean_popQA.csv \
#     --hidden-scorer-path=models/lama_hidden_scorer/tiiuae/falcon-7b-instruct/lama3_hscorer_0804_$i.pt \
#     --save-path=./results_pik2/tiiuae/falcon-7b-instruct/ \
#     --batch-size=32 \
#     --log-path=./logs/pt/falcon-7b-instruct/
#     wait

#     python src/self_knowledge/graphing/draw_graphs.py
#   done
# done
