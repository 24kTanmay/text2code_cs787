import torch
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- SETTINGS ---
# Points to your local model folder
model_name = "./t5_paraphraser_model" 
# Number of records to process
RECORDS_TO_PROCESS = 10000000
# -----------------

def paraphrase_text(text, tokenizer, model, device):
    """
    Processes a single string of text.
    """
    try:
        # 1. Prepare for T5
        text = f"paraphrase: {text}"
        
        # 2. Tokenize and run model (all on this worker's assigned GPU)
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=128,  # Truncate long docstrings
                truncation=True,
                padding=True
            ).to(device)
            
            outputs = model.generate(
                inputs["input_ids"],
                num_beams=5,
                num_return_sequences=1,
                max_length=128, # Set max_length for generation
                early_stopping=True
            )
            
            # 3. Decode the paraphrased output
            paraphrased_docstring = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        
        return paraphrased_docstring

    except Exception as e:
        print(f"ERROR: Could not paraphrase text '{text[:50]}...'. Error: {e}")
        return None # Return None on failure

def main(args):
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # --- Single-Threaded Setup ---
    print("Running in single-threaded CPU mode.")
    device = torch.device("cpu")

    # 1. Load Tokenizer and Model
    print(f"Loading model '{model_name}' onto CPU...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        print(f"CRITICAL: Failed to load model from cache. Error: {e}")
        return
    
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    # -------------------------

    # Read all lines from the input file
    print(f"Reading first {RECORDS_TO_PROCESS} lines from {args.input_file}...")
    lines = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= RECORDS_TO_PROCESS:
                break
            lines.append(line)
    
    print(f"Loaded {len(lines)} records.")

    # Open the output file
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        
        # Loop through the 100 lines with a progress bar
        for line in tqdm(lines, desc="Paraphrasing records"):
            try:
                # 1. Load the original record
                record = json.loads(line)
                docstring = record.get("docstring", "")
                
                # Write the original line to the file
                f_out.write(line.strip() + "\n")

                if not docstring:
                    # If docstring is empty, skip paraphrasing
                    continue

                # 2. Paraphrase the docstring
                paraphrased_doc = paraphrase_text(docstring, tokenizer, model, device)
                
                if paraphrased_doc:
                    # 3. Create and write the new augmented record
                    new_record = record.copy()
                    new_record["docstring"] = paraphrased_doc
                    new_record["_id"] = str(record["_id"]) + "_aug"
                    f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"ERROR: Failed to process line: {line[:50]}... Error: {e}")
    
    print(f"Done. Augmented data for {RECORDS_TO_PROCESS} records saved to {args.output_file}.")

if __name__ == "__main__":
    # --- This is the correct argument parser for this script ---
    parser = argparse.ArgumentParser(description="Augment dataset via docstring paraphrasing (DEBUG version).")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .json file (e.g., data/python_data.json)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .json file (e.g., data_augmented/python_data_debug.json)")
    args = parser.parse_args()
    # -----------------------------------------------------------
    
    main(args)
