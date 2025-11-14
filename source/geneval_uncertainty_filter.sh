#!/usr/bin/env bash

LOCAL_DIR_GEN="../generation_files"

BS=16
dtype='fp32'
incremental=False
greedy=False
filter_uncertainty=False
generation_factor=2 # Default: generate 2x samples for filtering

while [ $# -gt 0 ]
do
    case "$1" in
        -mf|--model_family)
            model_family="$2"
            shift
            ;;
        -mp|--model_path)
            model_path="$2"
            shift
            ;;
		-incr|--incremental)
            incremental="$2"
            shift
            ;;
		-cgxp|--codegeex_path)
            codegeex_path="$2"
            shift
            ;;
		-dat|--dataset)
            dataset="$2"
            shift
            ;;
		-greedy|--greedy)
            greedy="$2"
            shift
            ;;
		# --- NEW ARGUMENTS ---
		-filt|--filter)
            filter_uncertainty="$2"
            shift
            ;;
		-gfac|--gen_factor)
            generation_factor="$2"
            shift
            ;;
		# ---------------------
		*)
        echo "Unknown argument $1"
        exit 3
        ;;
    esac
    shift
done


model_name=$(basename ${model_path})

# prefix
if [[ "$model_name" == *"-prefix-"* ]]; then
	prefix=True
else
	prefix=False
fi

# separation
if [[ "$model_name" == *"-full-"* ]]; then
	replicated_tokens_map=True
	echo ">>> FULL Separation <<<"
	data_path="${model_path}"
elif [[ "$model_name" == *"-partial-"* ]]; then
	replicated_tokens_map=True
	echo ">>> PARTIAL Separation <<<"
	data_path="${model_path}"
else
	replicated_tokens_map=False
	data_path="none"
fi

# dataset
if [[ "$dataset" == "humaneval" ]]; then
#	data_eval_file="${codegeex_path}/codegeex/benchmark/humaneval-x/python/data/humaneval_python.jsonl.gz"
	data_eval_file="${codegeex_path}/codegeex/benchmark/humaneval-x/python/data/humaneval_python_mini.jsonl.gz"
	eval_script="${codegeex_path}/scripts/evaluate_humaneval_x.sh"
elif [[ "$dataset" == "mbpp" ]]; then
#       data_eval_file="${codegeex_path}/mbpp_test.jsonl"
	data_eval_file="${codegeex_path}/mbpp_test_mini.jsonl"
	eval_script="${codegeex_path}/scripts/evaluate_${dataset}.sh"
else
	echo "Invalid dataset name"
	exit
fi


echo "${model_name} | prefix=${prefix} | replicated_tokens_map=${replicated_tokens_map} | incremental=${incremental} | data=${dataset} | filter=${filter_uncertainty}"

output_dir="${LOCAL_DIR_GEN}/${model_name}_${dataset}_incr${incremental}"


if [[ "$greedy" == True ]]; then
	k=0
	p=1.0
	temp=1.0
	num_return_sequences=1
	
	echo "--- Running Greedy Generation (pass@1) ---"
	python generation.py \
		--torch_dtype="${dtype}" \
		--dataset_file="${data_eval_file}" \
		--model_name_or_path="${model_path}" \
		--max_seq_length=1024 \
		--output_dir="${output_dir}" \
		--greedy \
		--num_return_sequences="${num_return_sequences}" \
		--temperature="${temp}" \
		--k="${k}" \
		--p="${p}" \
		--batch_size="${BS}" \
		--seed=42 \
		--prefix_lm="${prefix}" \
		--model_type="${model_family}" \
		--replicated_tokens_map="${replicated_tokens_map}" \
		--data_path="${data_path}" \
		--incremental="${incremental}"

	file_to_evaluate="${output_dir}/samples=${num_return_sequences}_${dtype}_bs=${BS}_t=${temp}_k=${k}_p=${p}.jsonl"

else
	# --- This is the Stochastic Sampling (pass@k) branch ---
	k=0
	p=0.8
	temp=0.95
	samples_to_keep=200 
	
	if [[ "$filter_uncertainty" == True ]]; then
	    echo "--- Running Generation WITH UNCERTAINTY FILTERING ---"
		# 1. Generate MORE samples
		num_return_sequences=$(($samples_to_keep * $generation_factor))
		echo "Generating ${num_return_sequences} samples for filtering..."
	else
	    echo "--- Running Standard Stochastic Generation ---"
		# 1. Generate standard number of samples
		num_return_sequences=${samples_to_keep}
		echo "Generating ${num_return_sequences} samples..."
	fi

	generation_output_file="${output_dir}/samples=${num_return_sequences}_${dtype}_bs=${BS}_t=${temp}_k=${k}_p=${p}.jsonl"
	
	python generation.py \
		--torch_dtype="${dtype}" \
		--dataset_file="${data_eval_file}" \
		--model_name_or_path="${model_path}" \
		--max_seq_length=1024 \
		--output_dir="${output_dir}" \
		--num_return_sequences="${num_return_sequences}" \
		--temperature="${temp}" \
		--k="${k}" \
		--p="${p}" \
		--batch_size="${BS}" \
		--seed=42 \
		--prefix_lm="${prefix}" \
		--model_type="${model_family}" \
		--replicated_tokens_map="${replicated_tokens_map}" \
		--data_path="${data_path}" \
		--incremental="${incremental}"

	if [[ "$filter_uncertainty" == True ]]; then
		echo "--- Filtering generated samples from ${num_return_sequences} down to ${samples_to_keep} ---"
		file_to_evaluate="${output_dir}/samples=filtered_${samples_to_keep}_${dtype}_bs=${BS}_t=${temp}_k=${k}_p=${p}.jsonl"
		
		# 2. Run the new filter script
		python filter_generations.py \
			-i "${generation_output_file}" \
			-o "${file_to_evaluate}" \
			-k ${samples_to_keep}
		
		echo "Filtering complete. Filtered samples saved to ${file_to_evaluate}"
	else
		# 2. No filtering, evaluate the generated file directly
		file_to_evaluate="${generation_output_file}"
	fi
fi


# 3. Evaluate the final .jsonl file
echo "--- Evaluating file: ${file_to_evaluate} ---"
bash ${eval_script} \
"${file_to_evaluate}" \
"python" \
6 > "${file_to_evaluate%.jsonl}.out"

echo "Evaluation complete. Results saved to ${file_to_evaluate%.jsonl}.out"
