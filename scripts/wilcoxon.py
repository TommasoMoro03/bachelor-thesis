import re
import numpy as np
from scipy import stats

# Extract strategy names from the header line (assuming the structure)
# These are the ones where we will extract scores for.
strategy_names_raw = [
    "10-Sentence Chunking (1-Overlap)", "5-Sentence Chunking (1-Overlap)",
    "Fixed Size 1024/100", "Fixed Size 128/10", "Fixed Size 256/20",
    "Fixed Size 512/0", "Fixed Size 512/50", "High cohesion", "Higher buffer",
    "Moderate Cohesion", "Pure paragraphs",
    "Sentence Window (200-500 Chars, 50-Overlap)",
    "Sentence Window (500-1000 Chars, 100-Overlap)"
]

# Map these to the names used in the summary table if different (e.g. 'Mod Cohesion')
# For now, let's assume these are the exact names used as keys in the parsed data if needed
# But for statistical test, we just need the column data.

# Parse the raw data into a structured format
parsed_data = []
lines = raw_data.strip().split('\n')
data_lines = lines[1:]  # Skip the header line

for line in data_lines:
    row_dict = {}

    # Extract Experiment ID
    match_id = re.match(r'^\d+', line)
    exp_id = int(match_id.group(0))
    row_dict['Experiment ID'] = exp_id

    remaining_line = line[len(str(exp_id)):]

    # Find all numeric patterns (NDCG scores) in the rest of the line
    all_numbers_in_line = re.findall(r'(\d+\.\d+)', remaining_line)

    # The last 16 numbers are the scores
    ndcg_scores_str = all_numbers_in_line[-16:]
    ndcg_scores_float = [float(s) for s in ndcg_scores_str]

    # The text before the first score is the combined Doc Title and Question
    first_score_start_idx = remaining_line.find(ndcg_scores_str[0])
    doc_question_combined = remaining_line[:first_score_start_idx].strip()
    row_dict['Document & Question'] = doc_question_combined

    # Map scores to strategy names for easier access
    # There are 12 strategies.
    strategy_ndcg_scores = ndcg_scores_float[:12]

    for i, strategy_name in enumerate(strategy_names_raw):
        row_dict[strategy_name] = strategy_ndcg_scores[i]

    # Also store aggregate scores (though not directly used for Wilcoxon, good for completeness)
    row_dict['Average'] = ndcg_scores_float[12]
    row_dict['Std. Dev.'] = ndcg_scores_float[13]
    row_dict['Median'] = ndcg_scores_float[14]
    row_dict['Trimmed Mean'] = ndcg_scores_float[15]

    parsed_data.append(row_dict)

# Now, `parsed_data` is a list of dictionaries, where each dict is a row.
# We can extract lists of scores for each strategy.
strategy_data_for_wilcoxon = {strategy_name: [] for strategy_name in strategy_names_raw}

for row in parsed_data:
    for strategy_name in strategy_names_raw:
        score = row.get(strategy_name)
        if score is not None:  # Only include if a score exists for that strategy
            strategy_data_for_wilcoxon[strategy_name].append(score)

# --- Perform Wilcoxon Signed-Rank Tests ---
# We need to choose pairs for comparison.
# Let's pick some meaningful comparisons based on the overall summary table (which is what the user wants to analyze).
# From the summary_strategies.jpg, the strategies ordered by Mean NDCG are:
# 1. Pure paragraphs (0.4495)
# 2. Sentence Window (200-500 Chars, 50-Overlap) (0.4226)
# 3. Sentence Window (500-1000 Chars, 100-Overlap) (0.3861)
# 4. 5-Sentence Chunking (1-Overlap) (0.3770)
# 5. Fixed Size 128/10 (0.3610)
# 6. 10-Sentence Chunking (1-Overlap) (0.3582)
# 7. Moderate Cohesion (0.3561)
# 8. Fixed Size 256/20 (0.3537)
# 9. Fixed Size 512/0 (0.3172)
# 10. Fixed Size 512/50 (0.3043)
# 11. High cohesion (0.2871)
# 12. Higher buffer (0.2766)
# 13. Fixed Size 1024/100 (0.2357)

# Let's define the comparisons:
# 1. Best vs. Second Best: Pure paragraphs vs. SW (200-500)
# 2. Best vs. a mid-range fixed size: Pure paragraphs vs. Fixed Size 256/20
# 3. Best vs. a worst semantic: Pure paragraphs vs. Higher buffer
# 4. One good fixed size vs. a bad fixed size: Fixed Size 128/10 vs. Fixed Size 1024/100
# 5. One good structure vs. one good semantic: SW (200-500) vs. Moderate Cohesion

comparison_pairs = [
    ("Pure paragraphs", "Sentence Window (200-500 Chars, 50-Overlap)"),
    ("Pure paragraphs", "Fixed Size 256/20"),
    ("Pure paragraphs", "Higher buffer"),
    ("Fixed Size 128/10", "Fixed Size 1024/100"),
    ("Sentence Window (200-500 Chars, 50-Overlap)", "Moderate Cohesion")
]

wilcoxon_results = []
alpha = 0.05  # Significance level

print("\n--- Wilcoxon Signed-Rank Test Results ---")
for s1_name, s2_name in comparison_pairs:
    data1 = np.array(strategy_data_for_wilcoxon[s1_name])
    data2 = np.array(strategy_data_for_wilcoxon[s2_name])

    # The Wilcoxon test requires data of the same length
    if len(data1) != len(data2):
        print(f"Warning: Data lengths mismatch for {s1_name} ({len(data1)}) and {s2_name} ({len(data2)}). Skipping.")
        continue

    # Wilcoxon signed-rank test. `zero_method='wilcox'` is common.
    # `relational_method=True` is for older scipy versions if needed.
    # For newer scipy (>=1.7), `alternative='two-sided'` is default.
    # We are testing if the median difference is non-zero.

    # If the exact scores lead to zero differences, need to handle
    # `scipy.stats.wilcoxon` can return nan p-values if all differences are zero.
    diffs = data1 - data2
    if np.all(diffs == 0):
        statistic, p_value = 0.0, 1.0  # No difference observed
    else:
        statistic, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')  # Test for difference in either direction

    wilcoxon_results.append({
        "strategy_1": s1_name,
        "strategy_2": s2_name,
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < alpha
    })

    print(f"Comparing: '{s1_name}' vs. '{s2_name}'")
    print(f"  Wilcoxon Statistic: {statistic:.4f}, P-value: {p_value:.4f}")
    print(f"  Difference is statistically significant (p < {alpha}): {p_value < alpha}")

print("\n--- Summary of Wilcoxon Tests ---")
for result in wilcoxon_results:
    sig_status = "SIGNIFICANT" if result['significant'] else "NOT significant"
    print(f"'{result['strategy_1']}' vs. '{result['strategy_2']}': P={result['p_value']:.4f} ({sig_status})")