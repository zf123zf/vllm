model_path=/workspace/models/Qwen/Qwen2.5-Math-RM-72B
# model_path=/workspace/models/Qwen/Qwen2.5-72B-Instruct
echo model_path: $model_path
# CUDA_VISIBLE_DEVICES=2,3,4,5 python3 vllm/entrypoints/openai/api_server.py  --model ${model_path}  \
# 																			--served-model-name 72B \
# 																			--pipeline-parallel-size 1 \
# 								                                            --tensor-parallel-size 4 \
# 																			--gpu-memory-utilization 0.6 \
# 																			--max-model-len 363 2>&1 > z 


CUDA_VISIBLE_DEVICES=2,3,4,5 python3 vllm/entrypoints/openai/api_server.py  --model ${model_path}  \
																			--trust-remote-code \
																			--served-model-name Qwen2.5-Math-RM-72B \
																			--pipeline-parallel-size 1 \
								                                            --tensor-parallel-size 4 \
																			--gpu-memory-utilization 0.6 \
																			--max-model-len 363 2>&1 > z 

# model_path=/workspace/models/Qwen/Qwen2-Math-1.5B
# echo model_path: $model_path
# CUDA_VISIBLE_DEVICES=2,3,4,5 python3 vllm/entrypoints/openai/api_server.py  --model ${model_path}  \
# 																			--served-model-name test \
# 																			--gpu-memory-utilization 0.7 \
# 																			--max-model-len 128 2>&1 > z 



