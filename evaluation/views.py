import json
import math

import numpy as np

from django.db.models import Prefetch
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.contrib import messages
from django.db import transaction

# Import models from other apps and this app
from corpus.models import SourceText, Question
from evaluation.service import relevant_chunks, helper
from experiments.models import Experiment, ChunkingStrategy, ChunkSet, Chunk
from .models import ExperimentChunkAnalysis, RankedRelevantChunk, RetrievalSimulation
from .service.helper import handle_run_simulation_and_rdsg


def evaluation_detail_view(request, experiment_pk, chunk_set_pk):
    """
    Main view for evaluation analysis: displays status and handles actions.
    STEP 3: Implements POST logic to save ranking and compute properties.
    """
    experiment = get_object_or_404(
        Experiment.objects.select_related('source_text', 'question'),
        pk=experiment_pk
    )
    chunk_set = get_object_or_404(
        ChunkSet.objects.select_related('strategy'),
        pk=chunk_set_pk,
        source_text=experiment.source_text
    )

    try:
        analysis, created_by_this_request = ExperimentChunkAnalysis.objects.get_or_create(
            experiment=experiment,
            chunk_set=chunk_set
        )
    except Exception as e:
        messages.error(request, f"Error while accessing/creating the analysis: {e}")
        return redirect(reverse('dashboard'))

    if created_by_this_request:
        messages.info(request, f"Analysis record (ID: {analysis.id}) created.")
        try:
            print(f"Calling services.initialize_analysis for Analysis ID: {analysis.id}")
            relevant_chunks.initialize_analysis(analysis)
            analysis.refresh_from_db()  # Reload to update k_relevant
            messages.success(request, f"Initial analysis completed: {analysis.k_relevant} relevant chunks identified.")
        except Exception as e:
            messages.error(request, f"Error during initial analysis: {e}")
            print(f"Error in initialize_analysis: {e}")

    # --- Modular POST handling ---
    if request.method == 'POST':
        action = request.POST.get('action')
        # Flag to determine whether the page should be re-rendered (with errors)
        # or redirected (if POST operation was successful)
        post_action_failed_or_needs_form_rerender = False

        if action == 'save_ranking_and_calculate_properties':
            post_action_failed_or_needs_form_rerender = helper.handle_save_ranking_and_properties(
                request, analysis, experiment_pk, chunk_set_pk
            )
        elif action == 'run_simulation_and_calculate_rdsg':
            # --- CALL THE NEW HELPER FUNCTION ---
            post_action_failed_or_needs_form_rerender = handle_run_simulation_and_rdsg(
                request, analysis, experiment_pk, chunk_set_pk
            )
        else:
            messages.warning(request, "Unrecognized POST action.")
            post_action_failed_or_needs_form_rerender = True

        if not post_action_failed_or_needs_form_rerender:
            # If helper returns False (success), perform redirect
            return redirect(reverse('evaluation:detail', kwargs={'experiment_pk': experiment_pk, 'chunk_set_pk': chunk_set_pk}))
        # If True, the page is re-rendered and error messages are shown

    # --- Prepare context for GET (or failed POST) ---
    # (Logic for ranked_relevant_chunks, simulation, flags as before)
    ranked_relevant_chunks = analysis.ranked_relevant_chunks.select_related('chunk').order_by('ideal_rank', 'chunk__chunk_index')
    simulation = analysis.simulations.order_by('-ran_at').first()

    # These are the "ground truth" highlights for the *entire* document
    relevant_sentences = analysis.experiment.relevant_sentences.all().order_by('start_char')
    # Convert them to a simple list of {start, end} for JavaScript
    all_relevant_highlights_data = [
        {"start": sent.start_char, "end": sent.end_char}
        for sent in relevant_sentences
    ]

    retrieved_chunks_with_relevance = []
    relevant_chunk_map = {rrc.chunk_id: rrc for rrc in ranked_relevant_chunks}

    if simulation:
        retrieved_chunks_qs = simulation.retrieved_chunks.select_related('chunk').all()
        for retrieved_chunk_obj in retrieved_chunks_qs:
            relevance_info = relevant_chunk_map.get(retrieved_chunk_obj.chunk_id)
            retrieved_chunks_with_relevance.append({
                'retrieved_rank': retrieved_chunk_obj.retrieved_rank,
                'similarity_score_s': retrieved_chunk_obj.similarity_score_s,
                'chunk': retrieved_chunk_obj.chunk,
                'relevance_info': relevance_info
            })

    ranking_complete = False
    properties_calculated = False
    if analysis.k_relevant is not None and analysis.k_relevant > 0:
        # Consider only relevant chunks for this analysis
        ranked_count = RankedRelevantChunk.objects.filter(analysis=analysis).exclude(ideal_rank__isnull=True).count()
        if ranked_count == analysis.k_relevant:  # All relevant chunks are ranked
            ranking_complete = True
        else:
            if ranked_count > 0 and ranked_count < analysis.k_relevant:
                print(f"WARN: Incomplete ranking. {ranked_count}/{analysis.k_relevant} ranks assigned.")

        if ranking_complete:
            first_ranked_rrc = RankedRelevantChunk.objects.filter(analysis=analysis, ideal_rank__isnull=False).order_by('ideal_rank').first()
            if first_ranked_rrc and first_ranked_rrc.effective_relevance_w_prime is not None:
                properties_calculated = True

    print("PASSAGGIO DATI")
    print(json.dumps(all_relevant_highlights_data))

    context = {
        'experiment': experiment,
        'chunk_set': chunk_set,
        'analysis': analysis,
        'analysis_just_created': created_by_this_request and analysis.k_relevant is None,
        'ranked_relevant_chunks': ranked_relevant_chunks,
        'simulation': simulation,
        'retrieved_chunks_list': retrieved_chunks_with_relevance,
        'ranking_complete': ranking_complete,
        'properties_calculated': properties_calculated,
        'all_relevant_highlights_json': json.dumps(all_relevant_highlights_data),
    }
    return render(request, 'evaluation/evaluation_detail.html', context)


def view_evaluation_results(request):
    """
    Displays a summary table of NDCG scores for all experiments and chunking strategies.
    Allows for easy comparison across different strategies, including aggregate metrics.
    """
    print("Loading Evaluation Results Summary...")

    all_strategies = ChunkingStrategy.objects.all().order_by('name')

    experiments = Experiment.objects.select_related('source_text', 'question').order_by('source_text__title',
                                                                                        'question__text')

    results_data = []  # Data for the main table (per experiment row)

    # --- Aggregated Data Structures for the new summary table ---
    # Dictionary to store all NDCG scores for each strategy across all experiments
    # { 'Strategy Name A': [score1, score2, ...], 'Strategy Name B': [score1, score2, ...] }
    all_ndcg_scores_per_strategy = {strategy.name: [] for strategy in all_strategies}

    # Dictionary to store total ranking points for each strategy
    # { 'Strategy Name A': 120, 'Strategy Name B': 85 }
    strategy_ranking_points = {strategy.name: 0 for strategy in all_strategies}

    # Dictionary to store count of wins for each strategy
    # { 'Strategy Name A': 5, 'Strategy Name B': 2 }
    strategy_wins_count = {strategy.name: 0 for strategy in all_strategies}

    for experiment in experiments:
        row_data = {
            'experiment_id': experiment.id,
            'document_title': experiment.source_text.title,
            'question_text': experiment.question.text,
            'strategy_results': {},  # Stores {strategy_name: ndcg_score for current experiment}
            'ndcg_scores_for_calc': []  # Temporary list to collect scores for average/variance for current experiment
        }

        analyses_for_experiment = ExperimentChunkAnalysis.objects.filter(experiment=experiment).select_related(
            'chunk_set__strategy'
        ).prefetch_related(
            Prefetch(
                'simulations',
                queryset=RetrievalSimulation.objects.order_by('-ran_at'),
                to_attr='latest_simulations'
            )
        ).order_by('chunk_set__strategy__name')

        # Collect NDCG scores for this specific experiment (for internal ranking and per-row average/variance)
        current_experiment_strategy_ndcg_scores = {}  # {strategy_name: ndcg_score}

        for analysis in analyses_for_experiment:
            strategy_name = analysis.chunk_set.strategy.name
            ndcg_score = None
            if analysis.latest_simulations:
                latest_sim = analysis.latest_simulations[0]
                # Ensure ndcg_score is not None and not NaN/Inf for calculations
                if latest_sim.ndcg_score is not None and not math.isnan(latest_sim.ndcg_score) and not math.isinf(
                        latest_sim.ndcg_score):
                    ndcg_score = latest_sim.ndcg_score
                else:
                    print(
                        f"WARN: NDCG score for Experiment {experiment.id}, Strategy {strategy_name} is None/NaN/Inf: {latest_sim.ndcg_score}")

            row_data['strategy_results'][strategy_name] = ndcg_score
            if ndcg_score is not None:
                row_data['ndcg_scores_for_calc'].append(ndcg_score)
                current_experiment_strategy_ndcg_scores[strategy_name] = ndcg_score
                all_ndcg_scores_per_strategy[strategy_name].append(ndcg_score)  # Add to global list for aggregate calc

        # --- Calculate Average, Variance, Best/Worst Scores for the current row (Experiment) ---
        if row_data['ndcg_scores_for_calc']:
            scores_array = np.array(row_data['ndcg_scores_for_calc'])

            row_data['average_ndcg'] = np.mean(scores_array)
            # Use ddof=1 for sample standard deviation (unbiased estimator)
            row_data['std_dev_ndcg'] = np.std(scores_array, ddof=1) if len(scores_array) > 1 else 0.0
            row_data['median_ndcg'] = np.median(scores_array)

            # Trimmed Mean: Exclude top 1 and bottom 1 (if sufficient data)
            if len(scores_array) >= 3:  # Need at least 3 points to trim 1 from each end
                trimmed_scores_array = np.sort(scores_array)[1:-1]  # Remove first and last
                row_data['trimmed_mean_ndcg'] = np.mean(trimmed_scores_array)
            else:
                row_data['trimmed_mean_ndcg'] = None  # Not enough data to trim

            # Identify top 2 and bottom 2 scores for highlighting
            sorted_unique_scores_for_row = sorted(list(set(scores_array)), reverse=True)
            row_data['best_ndcg_score'] = sorted_unique_scores_for_row[0] if len(
                sorted_unique_scores_for_row) > 0 else None
            row_data['second_best_ndcg_score'] = sorted_unique_scores_for_row[1] if len(
                sorted_unique_scores_for_row) > 1 else None
            row_data['worst_ndcg_score'] = sorted_unique_scores_for_row[-1] if len(
                sorted_unique_scores_for_row) > 0 else None
            row_data['second_worst_ndcg_score'] = sorted_unique_scores_for_row[-2] if len(
                sorted_unique_scores_for_row) > 1 else None
        else:
            row_data['average_ndcg'] = None
            row_data['std_dev_ndcg'] = None
            row_data['median_ndcg'] = None
            row_data['trimmed_mean_ndcg'] = None
            row_data['best_ndcg_score'] = None
            row_data['second_best_ndcg_score'] = None
            row_data['worst_ndcg_score'] = None
            row_data['second_worst_ndcg_score'] = None

        # --- Calculate Ranking Points for the current experiment ---
        if current_experiment_strategy_ndcg_scores:
            # Sort strategies by their NDCG score for this experiment (descending)
            ranked_strategies_for_experiment = sorted(
                current_experiment_strategy_ndcg_scores.items(),
                key=lambda item: item[1] if item[1] is not None else -1,  # Sort None scores to the bottom
                reverse=True
            )

            points_per_rank = len(all_strategies)  # Start points from total number of strategies
            for rank, (strategy_name, score) in enumerate(ranked_strategies_for_experiment):
                if score is not None:  # Only assign points if a score was actually calculated
                    strategy_ranking_points[strategy_name] += (points_per_rank - rank)  # Assign points based on rank
                    if rank == 0:  # If it's the top rank (0-indexed)
                        strategy_wins_count[strategy_name] += 1
                else:  # Strategies with None scores get 0 points for this experiment
                    strategy_ranking_points[strategy_name] += 0  # Explicitly add 0

        results_data.append(row_data)

    # --- Calculate Aggregate Statistics for the Summary Table (Per Strategy) ---
    summary_table_data = []  # List of dicts, each dict represents a strategy's aggregate performance

    for strategy in all_strategies:
        strategy_name = strategy.name
        scores = all_ndcg_scores_per_strategy[strategy_name]

        agg_data = {
            'strategy_name': strategy_name,
            'mean_ndcg': None,
            'median_ndcg': None,
            'trimmed_mean_ndcg': None,
            'std_dev_ndcg': None,
            'total_ranking_points': strategy_ranking_points[strategy_name],
            'wins_count': strategy_wins_count[strategy_name],
            'num_experiments_run': len(scores)  # How many experiments this strategy actually participated in
        }

        if scores:
            scores_array = np.array(scores)
            agg_data['mean_ndcg'] = np.mean(scores_array)
            agg_data['median_ndcg'] = np.median(scores_array)
            agg_data['std_dev_ndcg'] = np.std(scores_array, ddof=1) if len(scores_array) > 1 else 0.0

            if len(scores_array) >= 3:
                # For trimmed mean, sort, remove the first and last (or more generally, a percentage)
                trimmed_scores_array = np.sort(scores_array)[1:-1]  # Exclude top 1 and bottom 1
                if len(trimmed_scores_array) > 0:
                    agg_data['trimmed_mean_ndcg'] = np.mean(trimmed_scores_array)
                else:
                    agg_data['trimmed_mean_ndcg'] = None
            else:
                agg_data['trimmed_mean_ndcg'] = None  # Not enough data for trimmed mean

        summary_table_data.append(agg_data)

    # Sort summary_table_data by a primary metric for presentation (e.g., Mean NDCG, descending)
    summary_table_data.sort(key=lambda x: x['mean_ndcg'] if x['mean_ndcg'] is not None else -1, reverse=True)

    context = {
        'all_strategies': all_strategies,
        'results_data': results_data,  # Data for the main per-experiment table
        'summary_table_data': summary_table_data,  # Data for the new aggregate summary table
    }
    return render(request, 'evaluation/results_summary.html', context)
