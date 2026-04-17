"""
Create question-answer tradeoff datasets using water-filling allocation.

Splits a fixed budget B equally across questions, varying the number of
answers per question to study More Questions vs More Answers (Section 4.1).

Two modes:
  - single: Split for one coverage ratio (e.g., 0.8) with specific answer counts
  - sub:    Split for multiple sub-coverage ratios extracted from a base coverage

Usage:
  python split_shortest_path_qa.py --mode single --coverage 0.8 --avg_num_answers 64
  python split_shortest_path_qa.py --mode sub --base_coverage 0.2 --sub_coverages 0.01,0.05,0.1
"""

import pickle
import os
import numpy as np
import random
import logging
import argparse


def water_filling_allocation(available_questions, question_caps, target_m, target_n, budget_B):
    """
    Allocate answers to questions using water-filling under caps.

    Args:
        available_questions: List of question keys
        question_caps: Dict mapping question -> max available answers (A_q)
        target_m: Target number of unique questions
        target_n: Target number of answers per question
        budget_B: Total budget of records

    Returns:
        selected_questions: List of selected question keys
        allocations: Dict mapping question -> number of answers to use
    """
    if target_m > len(available_questions):
        logging.warning(f"Target m={target_m} exceeds available questions {len(available_questions)}")
        target_m = len(available_questions)

    selected_questions = random.sample(available_questions, target_m)

    allocations = {}
    used_budget = 0
    for q in selected_questions:
        cap = question_caps[q]
        allocations[q] = min(target_n, cap)
        used_budget += allocations[q]

    remaining_budget = budget_B - used_budget

    while remaining_budget > 0:
        can_increment = [q for q in selected_questions if allocations[q] < question_caps[q]]

        if not can_increment:
            unused_questions = [q for q in available_questions if q not in selected_questions]
            if not unused_questions:
                logging.warning("All questions exhausted but budget not reached")
                break
            min_q = min(selected_questions, key=lambda x: allocations[x])
            max_unused_q = max(unused_questions, key=lambda x: question_caps[x])
            selected_questions.remove(min_q)
            remaining_budget += allocations[min_q]
            del allocations[min_q]
            selected_questions.append(max_unused_q)
            allocations[max_unused_q] = min(target_n, question_caps[max_unused_q])
            remaining_budget -= allocations[max_unused_q]
            continue

        increment_size = min(remaining_budget, len(can_increment))
        for i in range(increment_size):
            q = can_increment[i]
            allocations[q] += 1
            remaining_budget -= 1

        if remaining_budget == 0:
            break

    return selected_questions, allocations


def create_tradeoff_dataset(paths_file, avg_num_answers, budget_B, output_dir, seed=42):
    """
    Create dataset variants with different question/answer trade-offs.

    Args:
        paths_file: Path to the paths.pkl file
        avg_num_answers: List of target avg answers per question (e.g., [1,2,4,8,16,32,64])
        budget_B: Total budget of records
        output_dir: Directory to save the datasets
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    m_values = [int(budget_B // n) for n in avg_num_answers]

    with open(paths_file, 'rb') as f:
        all_paths = pickle.load(f)

    available_questions = list(all_paths.keys())
    question_caps = {q: len(paths) for q, paths in all_paths.items()}

    logging.info(f"Loaded {len(available_questions)} questions")
    logging.info(f"Question caps range: {min(question_caps.values())} to {max(question_caps.values())}")

    os.makedirs(output_dir, exist_ok=True)
    results_summary = []

    for idx, m in enumerate(m_values):
        target_n = avg_num_answers[idx]
        logging.info(f"\nProcessing m={m} (target avg answers: {budget_B/m:.2f})")

        selected_questions, allocations = water_filling_allocation(
            available_questions, question_caps, m, target_n, budget_B)

        dataset = {}
        total_records = 0
        for q in selected_questions:
            num_answers = allocations[q]
            available_paths = all_paths[q]
            if num_answers <= len(available_paths):
                selected_paths = random.sample(available_paths, num_answers)
            else:
                selected_paths = available_paths
            dataset[q] = selected_paths
            total_records += len(selected_paths)

        with open(os.path.join(output_dir, f'paths_ans{target_n}_B{budget_B}.pkl'), 'wb') as f:
            pickle.dump(dataset, f)

        allocation_data = {
            'selected_pairs': selected_questions,
            'num_paths_per_pair': allocations,
            'total_records': total_records,
            'target_m': m, 'target_n': target_n,
            'avg_num_answers': target_n, 'budget_B': budget_B,
        }
        with open(os.path.join(output_dir, f'pairs_ans{target_n}_B{budget_B}.pkl'), 'wb') as f:
            pickle.dump(allocation_data, f)

        actual_m = len(selected_questions)
        avg_answers = total_records / actual_m if actual_m > 0 else 0
        summary = {
            'target_m': m, 'actual_m': actual_m, 'target_n': target_n,
            'total_records': total_records, 'avg_answers': avg_answers,
            'min_answers': min(allocations.values()) if allocations else 0,
            'max_answers': max(allocations.values()) if allocations else 0,
            'budget_used': total_records, 'budget_target': budget_B
        }
        results_summary.append(summary)
        logging.info(f"  Questions: {actual_m}, Records: {total_records}, Avg answers: {avg_answers:.2f}")

    with open(os.path.join(output_dir, 'tradeoff_summary.pkl'), 'wb') as f:
        pickle.dump(results_summary, f)

    return results_summary


def analyze_tradeoff_results(summary_file):
    with open(summary_file, 'rb') as f:
        results = pickle.load(f)
    print("\nTrade-off Analysis:")
    print("=" * 60)
    print(f"{'Target M':<10} {'Actual M':<10} {'Total Rec':<10} {'Avg Ans':<10} {'Range':<15}")
    print("-" * 60)
    for r in results:
        range_str = f"[{r['min_answers']}-{r['max_answers']}]"
        print(f"{r['target_m']:<10} {r['actual_m']:<10} {r['total_records']:<10} "
              f"{r['avg_answers']:<10.2f} {range_str:<15}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Q/A tradeoff datasets with water-filling allocation.")
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'sub'])
    parser.add_argument('--coverage', type=float, default=0.8, help='Coverage ratio (single mode)')
    parser.add_argument('--avg_num_answers', type=str, default='64',
                        help='Comma-separated list of target avg answers per question (e.g., "1,2,4,8,16,32,64")')
    parser.add_argument('--base_coverage', type=float, default=0.2, help='Base coverage (sub mode)')
    parser.add_argument('--sub_coverages', type=str, default='0.01,0.05,0.1',
                        help='Comma-separated sub-coverage ratios (sub mode)')
    cli_args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    dataset_root = os.path.join(parent_dir, "dataset")

    avg_num_answers = [int(x) for x in cli_args.avg_num_answers.split(',')]

    if cli_args.mode == 'single':
        dataset_dir = os.path.join(dataset_root, "_spatial_length",
                                   f"coverage_ratio_{cli_args.coverage:.2f}", "pairs_0", "shortest_path")
        paths_file = os.path.join(dataset_dir, 'paths.pkl')
        output_dir = os.path.join(dataset_dir, 'tradeoff_datasets')

        log_file = os.path.join(output_dir, 'tradeoff_creation.log')
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

        with open(paths_file, 'rb') as f:
            all_paths = pickle.load(f)
        budget_B = len(all_paths)
        logging.info(f"Budget B = {budget_B}")

        summary = create_tradeoff_dataset(paths_file, avg_num_answers, budget_B, output_dir)
        analyze_tradeoff_results(os.path.join(output_dir, 'tradeoff_summary.pkl'))

    elif cli_args.mode == 'sub':
        with open(os.path.join(dataset_root, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
            indices_to_nodes = pickle.load(f)

        base_dir = os.path.join(dataset_root, "_spatial_length",
                                f"coverage_ratio_{cli_args.base_coverage:.2f}", "pairs_0", "shortest_path")
        with open(os.path.join(base_dir, 'paths.pkl'), 'rb') as f:
            all_paths = pickle.load(f)

        sub_covs = [float(x) for x in cli_args.sub_coverages.split(',')]
        for sub_cov in sub_covs:
            logging.info(f"Sub-coverage ratio: {sub_cov}")
            sub_folder = os.path.join(dataset_root, "_spatial_length",
                                      f"coverage_ratio_{sub_cov:.2f}", "pairs_0")
            sub_paths_file = os.path.join(sub_folder, 'shortest_path', 'paths.pkl')
            output_dir = os.path.join(sub_folder, 'shortest_path', 'tradeoff_datasets')

            if os.path.exists(sub_paths_file):
                with open(sub_paths_file, 'rb') as f:
                    paths = pickle.load(f)
            else:
                with open(os.path.join(sub_folder, 'sample_pairs.pkl'), 'rb') as f:
                    sub_pairs = pickle.load(f)
                os.makedirs(os.path.join(sub_folder, 'shortest_path'), exist_ok=True)
                paths = {}
                for pair in sub_pairs:
                    start_idx, end_idx = pair
                    start = indices_to_nodes[start_idx]
                    end = indices_to_nodes[end_idx]
                    paths[(start, end)] = all_paths[(start, end)]
                with open(sub_paths_file, 'wb') as f:
                    pickle.dump(paths, f)

            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, 'tradeoff_creation.log')
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

            budget_B = len(paths)
            logging.info(f"Budget B = {budget_B}")

            summary = create_tradeoff_dataset(sub_paths_file, avg_num_answers, budget_B, output_dir)
            analyze_tradeoff_results(os.path.join(output_dir, 'tradeoff_summary.pkl'))
