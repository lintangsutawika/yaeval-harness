# Yet Another Evaluation Harness

# Usage

### Launching VLLM API Model

You will need to launch a model first before running evaluations.
```
vllm serve meta-llama/Llama-3.1-8B-Instruct > tmp.txt &
```

### Run Evaluations

```
yeval \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --sample_args "temperature=0.6,top_p=0.9,n=10" \
    --task commonsense_qa \
    --run_name $RUNNAME \
    --trust_remote_code \
    --output_path ./output_dir/
```

You can also change system messages with `--system_message`

```
--system_message "Before answering, think step-by-step"
```
