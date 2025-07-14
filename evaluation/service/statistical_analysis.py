import numpy as np
from scipy import stats
from django.db.models import Prefetch  #

# Import Django Models from your project
from experiments.models import ChunkingStrategy, Experiment  #
from evaluation.models import ExperimentChunkAnalysis, RetrievalSimulation  #


def get_ndcg_scores_per_strategy():
    """
    Fetches NDCG scores for all strategies across all experiments from the database.
    Returns a dictionary: {strategy_name: [ndcg_score_exp1, ndcg_score_exp2, ...]}
    """
    print("Fetching NDCG scores from database for statistical analysis...")

    # Get all chunking strategies, ordered by name
    all_strategies = ChunkingStrategy.objects.all().order_by('name')  #
    # Get all unique experiments
    experiments = Experiment.objects.all()  #

    # Initialize a dictionary to store lists of NDCG scores for each strategy
    ndcg_scores_per_strategy = {strategy.name: [] for strategy in all_strategies}  #

    # Iterate through each experiment to collect scores
    for experiment in experiments:  #
        # Prefetch analyses and their latest simulations for this specific experiment for efficiency
        analyses_for_experiment = ExperimentChunkAnalysis.objects.filter(experiment=experiment).select_related(
            'chunk_set__strategy'  #
        ).prefetch_related(
            Prefetch(
                'simulations',  # Related name for RetrievalSimulation in ExperimentChunkAnalysis
                queryset=RetrievalSimulation.objects.order_by('-ran_at'),  # Get the most recent simulation
                to_attr='latest_simulations'  # Assigns the prefetched simulations to this attribute
            )
        ).order_by('chunk_set__strategy__name')  # Ensure consistent ordering of strategies within an experiment

        # Create a temporary dictionary to hold scores for the current experiment across all strategies.
        # This is crucial to ensure paired data for Wilcoxon, by filling missing scores with None.
        current_experiment_ndcg_scores = {strategy.name: None for strategy in all_strategies}  #

        for analysis in analyses_for_experiment:  #
            strategy_name = analysis.chunk_set.strategy.name  #
            if analysis.latest_simulations:  #
                latest_sim = analysis.latest_simulations[0]  #
                # Validate NDCG score before adding: must not be None, NaN, or Inf
                if latest_sim.ndcg_score is not None and not np.isnan(latest_sim.ndcg_score) and not np.isinf(
                        latest_sim.ndcg_score):  #
                    current_experiment_ndcg_scores[strategy_name] = latest_sim.ndcg_score  #
                else:
                    print(
                        f"WARN: NDCG score for Experiment {experiment.id}, Strategy {strategy_name} is invalid (None/NaN/Inf): {latest_sim.ndcg_score}")  #

        # Append scores from the current experiment to the main `ndcg_scores_per_strategy` structure.
        # This ensures each strategy's list has an entry (score or None) for every experiment.
        for strategy_name_obj in all_strategies:  # Iterate through strategy objects to ensure all are covered
            strategy_name = strategy_name_obj.name  #
            ndcg_scores_per_strategy[strategy_name].append(current_experiment_ndcg_scores[strategy_name])  #

    return ndcg_scores_per_strategy  #


def run_wilcoxon_tests(ndcg_scores_data, comparison_pairs, alpha=0.05):
    """
    Performs Wilcoxon Signed-Rank tests for specified pairs of strategies.

    Args:
        ndcg_scores_data (dict): Dictionary {strategy_name: [ndcg_score_exp1, ...]}
        comparison_pairs (list): List of tuples, each tuple is (strategy_name_1, strategy_name_2)
        alpha (float): Significance level for the p-value.

    Returns:
        list: List of dictionaries, each containing test results for a comparison pair.
    """
    wilcoxon_results = []  #

    print("\n--- Performing Wilcoxon Signed-Rank Tests ---")  #
    for s1_name, s2_name in comparison_pairs:  #
        # Get data for the two strategies being compared
        data1 = np.array(ndcg_scores_data.get(s1_name, []), dtype=float)  # Ensure numpy array of floats
        data2 = np.array(ndcg_scores_data.get(s2_name, []), dtype=float)  #

        # Filter out pairs where either data point is NaN (missing simulation data for that experiment)
        valid_indices = ~np.isnan(data1) & ~np.isnan(data2)  #

        filtered_data1 = data1[valid_indices]  #
        filtered_data2 = data2[valid_indices]  #

        num_valid_experiments = len(filtered_data1)  #

        # Handle edge cases for Wilcoxon test
        if num_valid_experiments < 2:  # Wilcoxon needs at least 2 non-zero differences for a meaningful test
            statistic, p_value = np.nan, np.nan  #
            significant = False  #
            print(
                f"Skipping comparison '{s1_name}' vs. '{s2_name}': Not enough valid paired data ({num_valid_experiments} experiments).")  #
        elif np.all(filtered_data1 == filtered_data2):  # All differences are exactly zero
            statistic, p_value = 0.0, 1.0  #
            significant = False  #
        else:
            try:
                # Perform the Wilcoxon signed-rank test
                # alternative='two-sided' checks for a difference in either direction.
                # zero_method='wilcox' handles zero differences as recommended.
                statistic, p_value = stats.wilcoxon(filtered_data1, filtered_data2, alternative='two-sided',
                                                    zero_method='wilcox')  #
                significant = p_value < alpha  #
            except ValueError as e:
                # Catch specific ValueError if test cannot be performed (e.g., all differences are zero
                # or only one non-zero difference after filtering).
                print(f"Warning: ValueError in Wilcoxon test for '{s1_name}' vs '{s2_name}': {e}")  #
                statistic, p_value = np.nan, np.nan  #
                significant = False  #

        # Calculate mean NDCG for each strategy in the comparison
        mean_s1 = np.mean(filtered_data1) if num_valid_experiments > 0 else np.nan  #
        mean_s2 = np.mean(filtered_data2) if num_valid_experiments > 0 else np.nan  #

        wilcoxon_results.append({
            "strategy_1": s1_name,  #
            "strategy_2": s2_name,  #
            "num_experiments": num_valid_experiments,  #
            "statistic": statistic,  #
            "p_value": p_value,  #
            "significant": significant,  #
            "mean_s1": mean_s1,  #
            "mean_s2": mean_s2  #
        })

        print(f"Comparing: '{s1_name}' vs. '{s2_name}' (N={num_valid_experiments} valid pairs)")  #
        print(f"  Wilcoxon Statistic: {statistic:.4f}, P-value: {p_value:.4f}")  #
        print(f"  Difference is statistically significant (p < {alpha}): {significant}")  #

    return wilcoxon_results  #