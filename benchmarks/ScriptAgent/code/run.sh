# Configuration
JSONL_FILE="./code/corecode/test_responses.jsonl"
OUTPUT_BASE="./output_comparison_1script_1217"
BASELINE_DIR="./output_comparison_1script_1217/veo3.1/node_video"

# Step 1: Generate Veo3.1 baseline (all scripts)
echo "🎬 Step 1: Generate Veo3.1 baseline..."
python ./code/corecode/director_agent.py \
    --model veo3.1 \
    --style anime \
    --output_dir "$OUTPUT_BASE" \
    --enable_batch \
    --batch_jsonl "$JSONL_FILE" \
    --seconds 8
