from django.contrib import messages
from django.db import transaction, IntegrityError
from evaluation.service import chunk_properties, retrieval_simulation

from evaluation.models import ExperimentChunkAnalysis, RankedRelevantChunk


def handle_save_ranking_and_properties(request, analysis: ExperimentChunkAnalysis, experiment_pk: int, chunk_set_pk: int):
    """
    Handles manual ranking submission and starts property calculation.
    Returns True if the page should be re-rendered due to an error,
    False if the operation was successful and redirect is allowed.
    """
    print(f"POST action: save_ranking_and_calculate_properties for Analysis ID: {analysis.id}")

    if analysis.k_relevant is None:
        messages.error(request, "Error: k_relevant is not defined for this analysis. Please run the initialization first.")
        return True  # Indicates error, forces re-render

    # 1. Retrieve and parse submitted ranks from the form
    submitted_ranks_map = {}  # {chunk_pk: rank_value}
    parse_error_occurred = False
    expected_chunk_pks_for_ranking = set(
        RankedRelevantChunk.objects.filter(analysis=analysis).values_list('chunk_id', flat=True)
    )

    if analysis.k_relevant > 0 and not expected_chunk_pks_for_ranking:
        messages.error(request, "Error: k_relevant > 0 but no RankedRelevantChunk found for this analysis.")
        return True

    found_chunk_pks_in_post = set()

    for key, value_str in request.POST.items():
        if key.startswith('rank_'):
            try:
                chunk_pk = int(key.split('_')[1])
                found_chunk_pks_in_post.add(chunk_pk)
                if not value_str:  # Empty rank field
                    messages.error(request, f"Error: Missing rank for chunk ID (PK) {chunk_pk}.")
                    parse_error_occurred = True
                    continue

                rank_value = int(value_str)

                if analysis.k_relevant > 0 and not (1 <= rank_value <= analysis.k_relevant):
                    messages.error(request, f"Error: Rank '{rank_value}' for chunk PK {chunk_pk} is out of range [1, {analysis.k_relevant}].")
                    parse_error_occurred = True
                elif chunk_pk in submitted_ranks_map:
                    messages.error(request, f"Error: Duplicate rank submitted for chunk PK {chunk_pk}.")
                    parse_error_occurred = True
                else:
                    submitted_ranks_map[chunk_pk] = rank_value

            except (ValueError, IndexError, TypeError):
                messages.error(request, f"Invalid rank format for '{key}': '{value_str}'. Please use integers.")
                parse_error_occurred = True

    if parse_error_occurred:
        return True  # Indicates error, forces re-render

    # 2. Logical validation of ranks
    validation_failed = False
    if analysis.k_relevant > 0:
        if expected_chunk_pks_for_ranking != found_chunk_pks_in_post:
            missing_ranks_for_pks = expected_chunk_pks_for_ranking - found_chunk_pks_in_post
            extra_pks_in_post = found_chunk_pks_in_post - expected_chunk_pks_for_ranking
            if missing_ranks_for_pks:
                messages.error(request, f"Error: Missing ranks for chunk PKs: {missing_ranks_for_pks}.")
            if extra_pks_in_post:
                messages.error(request, f"Error: Unexpected ranks received for chunk PKs: {extra_pks_in_post}.")
            validation_failed = True

        all_rank_values = list(submitted_ranks_map.values())
        if len(all_rank_values) != len(set(all_rank_values)):
            messages.error(request, "Error: Assigned ranks must be unique.")
            validation_failed = True
        else:
            expected_rank_sequence = set(range(1, analysis.k_relevant + 1))
            if set(all_rank_values) != expected_rank_sequence:
                messages.error(request, f"Error: Ranks must cover all values from 1 to {analysis.k_relevant} without duplicates or gaps. Received: {sorted(all_rank_values)}")
                validation_failed = True
    elif analysis.k_relevant == 0 and submitted_ranks_map:
        messages.warning(request, "k_relevant is 0, but ranks were submitted. They will be ignored.")

    if validation_failed:
        return True  # Indicates error, forces re-render

    # 3. If valid (or k_relevant == 0), proceed with saving and calculation
    try:
        with transaction.atomic():
            if analysis.k_relevant > 0:
                updated_rrcs = []
                for chunk_pk, rank_value in submitted_ranks_map.items():
                    try:
                        rrc = RankedRelevantChunk.objects.get(analysis=analysis, chunk_id=chunk_pk)
                        rrc.ideal_rank = rank_value
                        updated_rrcs.append(rrc)
                    except RankedRelevantChunk.DoesNotExist:
                        messages.error(request, f"Critical error: RankedRelevantChunk not found for chunk PK {chunk_pk}.")
                        raise

                if updated_rrcs:
                    RankedRelevantChunk.objects.bulk_update(updated_rrcs, ['ideal_rank'])
                    print(f"Updated {len(updated_rrcs)} ranks in the DB.")

            # Call the service to calculate chunk properties (w, Density, w′)
            chunk_properties.calculate_chunk_properties(analysis)

            messages.success(request, "Ranking saved and properties successfully calculated.")
            return False

    except IntegrityError as e:
        messages.error(request, f"Database error while saving ranks: {e}")
    except Exception as e:
        messages.error(request, f"Unexpected error while saving ranks or calculating properties: {e}")
        print(f"Unhandled exception in _handle_save_ranking_and_properties: {e}")

    return True


def handle_run_simulation_and_rdsg(request, analysis: ExperimentChunkAnalysis, experiment_pk: int, chunk_set_pk: int):
    """
    Handles the retrieval simulation run and RDSG score calculation.
    Returns True if an error occurs, False if successful.
    """
    print(f"POST action: run_simulation_and_calculate_rdsg for Analysis ID: {analysis.id}")

    # 1. Check prerequisites: chunk properties (w′) must have been calculated
    if analysis.k_relevant is None:
        messages.error(request, "Error: k_relevant is not defined. Please perform the initial analysis first.")
        return True

    if analysis.k_relevant > 0:
        properties_exist = RankedRelevantChunk.objects.filter(
            analysis=analysis,
            effective_relevance_w_prime__isnull=False
        ).exists()
        if not properties_exist:
            messages.error(request, "Error: Chunk properties (w′) must be calculated before running the simulation. Please save the ranking first.")
            return True

    # 2. Call services
    try:
        with transaction.atomic():
            simulation = retrieval_simulation.run_retrieval_simulation(analysis)

            if not simulation:
                messages.error(request, "Retrieval simulation creation failed.")
                return True

            messages.info(request, f"Simulation (ID: {simulation.id}) completed. Retrieved {simulation.k_retrieved} chunks.")

            # Compute RDSG
            retrieval_simulation.calculate_rdsg_and_ndcg(simulation)

            messages.success(request, f"NDCG score calculated: {simulation.ndcg_score:.4f}")
            return False

    except ValueError as ve:
        messages.error(request, f"Error during simulation preparation: {ve}")
    except Exception as e:
        messages.error(request, f"Unexpected error during simulation or RDSG calculation: {e}")
        print(f"Unhandled exception in _handle_run_simulation_and_rdsg: {e}")

    return True
