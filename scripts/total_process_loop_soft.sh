for perturb_times in 0; do
    for perturb_prob in 0.10; do
        echo "#####当前参数: perturb_times=$perturb_times, perturb_prob=$perturb_prob"

        # 在4张卡上同时运行4个模型，提取隐藏层并训练评分器
        models=("download_models/Qwen3-4B-Instruct-2507" "download_models/Meta-Llama-3-8B-Instruct" "download_models/Llama-2-7b-chat-hf" "download_models/falcon-7b-instruct")
        # models=("download_models/falcon-7b-instruct")

        gpus=(4 5 6 7)  # 4张卡的索引
        # gpus=(7)  # 4张卡的索引

        # 确保模型数量和GPU数量匹配
        if [ ${#models[@]} -ne ${#gpus[@]} ]; then
            echo "模型数量与GPU数量不匹配"
            exit 1
        fi

        # 循环启动每个模型，分配到不同的GPU
        for i in "${!models[@]}"; do
            model="${models[$i]}"
            gpu="${gpus[$i]}"
            layer=24  # 指定要提取的层索引
            
            echo "在GPU $gpu 上运行模型: $model，提取层: $layer"
            
            # 在后台运行，为每个进程分配独立的GPU
            (
                # 提取train隐藏层特征
                CUDA_VISIBLE_DEVICES=$gpu python -m self_knowledge.evaluation.extract_hidden_layers_temp \
                    --model-name="$model" \
                    --hidden-layer-index="$layer" \
                    --dataset-path="results_kn_triviaqa/$model/trivia_qa_sf_all_None.csv" \
                    --perturb-times="$perturb_times" \
                    --perturb-prob="$perturb_prob" \
                    --save-file="models/$model/lama3_triviaqa_soft_hlayer_$layer.pt"

                # # 提取validition隐藏层特征
                # CUDA_VISIBLE_DEVICES=$gpu python -m self_knowledge.evaluation.extract_hidden_layers_temp \
                #     --model-name="$model" \
                #     --hidden-layer-index="$layer" \
                #     --dataset-path="results_kn_triviaqa/$model/popqa_sf_1000_None.csv" \
                #     --save-file="models/$model/lama3_triviaqa_soft_hlayer_validation_$layer.pt"
                
                # 训练评分器
                CUDA_VISIBLE_DEVICES=$gpu python -m self_knowledge.evaluation.train_scorer_soft \
                    --hidden-path="models/$model/lama3_triviaqa_soft_hlayer_$layer.pt" \
                    --out-name="models/lama_hidden_scorer/$model/lama3_triviaqa_hscorer_soft_$layer.pt"
            ) &
        done

        # 等待所有后台进程完成
        wait
        echo "所有模型的隐藏层提取和评分器训练已完成"


        # 清空results_pik2_triviaqa
        rm -rf results_pik2_triviaqa_soft

        # # 运行 P(IK) 版本的 self-confidence 评估（popQA 数据集）
        models=("download_models/Qwen3-4B-Instruct-2507" "download_models/Meta-Llama-3-8B-Instruct" "download_models/Llama-2-7b-chat-hf" "download_models/falcon-7b-instruct")
        gpus=(4 5 6 7)  # 4张卡的索引

        # 确保模型数量和GPU数量匹配
        if [ ${#models[@]} -ne ${#gpus[@]} ]; then
            echo "错误：模型数量与GPU数量不匹配"
            exit 1
        fi

        # 循环启动每个模型，分配到不同的GPU
        for i in "${!models[@]}"; do
            model="${models[$i]}"
            gpu="${gpus[$i]}"
            
            echo "在GPU $gpu 上运行模型: $model"
            
            # 为每个模型创建独立的日志目录
            mkdir -p "logs/pt/$model"
            
            # # # 在后台运行，指定GPU并输出日志
            # (
            #     CUDA_VISIBLE_DEVICES=$gpu python src/self_knowledge/main_soft.py \
            #         --model-name="$model" \
            #         --dataset-path="data/trivia_qa/validation_nocontent.csv" \
            #         --hidden-scorer-path="models/lama_hidden_scorer/$model/lama3_triviaqa_hscorer_soft_24.pt" \
            #         --save-path="results_pik2_triviaqa_soft/$model/" \
            #         --batch-size=32 \
            #         --log-path="logs/pt/$model/"
            # ) &
            # # 在后台运行，指定GPU并输出日志
            (
                CUDA_VISIBLE_DEVICES=$gpu python src/self_knowledge/main_soft.py \
                    --model-name="$model" \
                    --dataset-path="results_kn_triviaqa/$model/validation_sf_None.csv" \
                    --hidden-scorer-path="models/lama_hidden_scorer/$model/lama3_triviaqa_hscorer_soft_24.pt" \
                    --save-path="results_pik2_triviaqa_soft/$model/" \
                    --batch-size=32 \
                    --log-path="logs/pt/$model/"
            ) &
            # ) > "logs/pt/$model/run.log" 2>&1 &
        done

        # 等待所有后台进程完成
        wait
        echo "所有模型的 P(IK) 评估已完成"
        echo "#####当前参数: perturb_times=$perturb_times, perturb_prob=$perturb_prob"


        python src/self_knowledge/graphing/draw_graphs.py \
            --results-path="results_pik2_triviaqa_soft/" \
            --force-check-labels-path="results_kn_triviaqa/" \
            --csv-filename="validation_sf_None.csv"

    done
done
