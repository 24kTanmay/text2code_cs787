import json
import argparse
import Levenshtein as lev
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import combinations
from collections import defaultdict

def group_by_task(jsonl_path):
    """Reads the generated samples and groups them by task_id."""
    tasks = defaultdict(list)
    print(f"Reading and grouping samples from {jsonl_path}...")
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            tasks[data['task_id']].append(data)
    print(f"Found {len(tasks)} unique tasks.")
    return tasks

def calc_distance_pair(pair):
    """
    Helper function to calculate Levenshtein distance between two strings.
    'pair' is a tuple of two (index, string) tuples:
    ( (index1, string1), (index2, string2) )
    """
    # Get the string from the first element of the pair
    string1 = pair[0][1] 
    # Get the string from the second element of the pair
    string2 = pair[1][1]
    
    return lev.distance(string1, string2)

def filter_task_group(task_data, num_to_keep, threshold_ratio=0.1):
    """
    Filters a group of generations for a single task down to the num_to_keep.
    Uses parallel processing to compute a distance matrix.
    """
    generations = [d['generation'] for d in task_data]
    num_generated = len(generations)
    
    if num_generated <= num_to_keep:
        return task_data
    
    # 1. Create all unique pairs of (index, generation_string)
    pairs = list(combinations(enumerate(generations), 2))
    
    # 2. Compute distances in parallel
    n_workers = max(1, cpu_count() // 2)
    with Pool(processes=n_workers) as pool:
        distances = list(tqdm(
            pool.imap(calc_distance_pair, pairs, chunksize=1000), 
            total=len(pairs), 
            desc=f"Filtering {task_data[0]['task_id']}", 
            leave=False
        ))

    # 3. Build the distance matrix
    dist_matrix = np.zeros((num_generated, num_generated), dtype=np.int32)

    # 'pairs' has elements like ((idx1, str1), (idx2, str2))
    # 'distances' has elements like (dist)
    # 'zip' creates an iterator of (( (idx1, str1), (idx2, str2) ), dist)
    for pair, dist in zip(pairs, distances):
        # pair[0] is (idx1, str1)
        # pair[1] is (idx2, str2)
        idx1 = pair[0][0]
        idx2 = pair[1][0]
        dist_matrix[idx1, idx2] = dist

    # 4. Compute agreement scores
    agreement_scores = np.zeros(num_generated)
    for i in range(num_generated):
        gen_len = max(1, len(generations[i])) # Avoid division by zero
        threshold = gen_len * threshold_ratio
        agreement_scores[i] = np.sum(dist_matrix[i] < threshold)
        
    # 5. Get the indices of the top N samples with the highest agreement
    indices_to_keep = np.argsort(agreement_scores)[-num_to_keep:]
    
    # Return the original data dictionaries for the survivors
    return [task_data[i] for i in indices_to_keep]


def main(args):
    # 1. Read and group all generated samples
    tasks = group_by_task(args.input_file)
    
    # 2. Filter each task group
    filtered_results = []
    
    for task_id in tqdm(tasks, desc="Overall Task Progress"):
        task_data = tasks[task_id]
        filtered_group = filter_task_group(task_data, args.keep)
        filtered_results.extend(filtered_group)

    # 3. Write the filtered list to the output file
    print(f"\nWriting {len(filtered_results)} filtered samples to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        for item in filtered_results:
            f.write(json.dumps(item) + "\n")
    
    print("Filtering complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter generated samples based on cluster agreement.")
    parser.add_argument("--input_file", "-i", type=str, required=True, help="Input .jsonl file with all generated samples.")
    parser.add_argument("--output_file", "-o", type=str, required=True, help="Output .jsonl file for filtered samples.")
    
    parser.add_argument("--keep", "-k", type=int, default=200, help="Number of samples to keep per task.")
    
    # Add spawn method for compatibility with multiprocessing + CUDA
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main(parser.parse_args())
