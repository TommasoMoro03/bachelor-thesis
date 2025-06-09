import json

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
    Allows for easy comparison across different strategies.
    """
    print("Loading Evaluation Results Summary...")

    # Fetch all Chunking Strategies for column headers
    all_strategies = ChunkingStrategy.objects.all().order_by('name')

    # Fetch all Experiments (Document-Question pairs)
    # We want to optimize queries by prefetching related data
    experiments = Experiment.objects.select_related('source_text', 'question').order_by('source_text__title', 'question__text')

    results_data = [] # List of dicts, each dict represents a row (an experiment)

    # Dictionary to quickly map strategy_id to its name
    strategy_name_map = {strategy.id: strategy.name for strategy in all_strategies}

    for experiment in experiments:
        row_data = {
            'experiment_id': experiment.id,
            'document_title': experiment.source_text.title,
            'question_text': experiment.question.text,
            'strategy_results': {} # Will store {strategy_name: ndcg_score}
        }

        # Fetch all analyses for this experiment, prefetching related chunk_set and simulations
        # Order by chunk_set__strategy__name to ensure consistent column order
        analyses_for_experiment = ExperimentChunkAnalysis.objects.filter(experiment=experiment).select_related(
            'chunk_set__strategy'
        ).prefetch_related(
            Prefetch(
                'simulations',
                queryset=RetrievalSimulation.objects.order_by('-ran_at'), # Get the latest simulation if multiple
                to_attr='latest_simulations' # Store in this attribute
            )
        ).order_by('chunk_set__strategy__name') # Sort analyses by strategy name for consistent display

        for analysis in analyses_for_experiment:
            strategy_name = analysis.chunk_set.strategy.name
            ndcg_score = None
            if analysis.latest_simulations:
                latest_sim = analysis.latest_simulations[0] # Get the most recent simulation
                ndcg_score = latest_sim.ndcg_score

            row_data['strategy_results'][strategy_name] = ndcg_score

        results_data.append(row_data)

    context = {
        'all_strategies': all_strategies, # For table headers
        'results_data': results_data,     # For table rows
    }
    return render(request, 'evaluation/results_summary.html', context)
