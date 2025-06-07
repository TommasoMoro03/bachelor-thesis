import json
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib import messages
from django.db import transaction, IntegrityError  # For atomic operations and DB error handling

from corpus.service import source_text_service
from experiments.models import Experiment, RelevantSentence
from corpus.models import SourceText  # Only import SourceText

# --- KEEP list_experiments view if needed ---
def list_experiments(request):
    """Display a list of all created experiments."""
    experiments = Experiment.objects.select_related('source_text', 'question').all()
    context = {'experiments': experiments}
    return render(request, 'experiments/list_experiments.html', context)


# --- MAIN VIEW FOR ANNOTATION (GET + POST) ---
@transaction.atomic  # Make the save/delete process atomic
def annotate_experiment_view(request, experiment_pk):
    """
    Handles both display (GET) and saving (POST) of annotations for an experiment.
    """
    experiment = get_object_or_404(
        Experiment.objects.select_related('source_text', 'question'),
        pk=experiment_pk
    )
    source_text_content = source_text_service.get_full_text(experiment.source_text).strip()

    if request.method == 'POST':
        # --- Logic to save new highlights ---
        highlights_json = request.POST.get('highlights', '[]')  # Get JSON from hidden input
        try:
            highlights_data = json.loads(highlights_json)
            if not isinstance(highlights_data, list):
                raise ValueError("Highlights format is not a valid JSON list.")

            # 1. Delete ALL previous annotations for this experiment
            #    This is because the JS sends the COMPLETE desired state of highlights.
            RelevantSentence.objects.filter(experiment=experiment).delete()

            # 2. Create new annotations based on the received data
            sentences_to_create = []
            processed_ranges = set()  # Avoid exact duplicates in the input JSON

            for item in highlights_data:
                if not isinstance(item, dict) or 'start' not in item or 'end' not in item:
                    messages.error(request, "Invalid highlight format (missing 'start' or 'end').")
                    return redirect(reverse('experiments:annotate_experiment', kwargs={'experiment_pk': experiment_pk}))

                start = int(item['start'])
                end = int(item['end'])
                range_tuple = (start, end)

                # Basic validation
                if start < 0 or end < 0 or start >= end:
                    messages.warning(request, f"Ignored invalid range: start={start}, end={end}")
                    continue

                # Avoid duplicate entries in the input
                if range_tuple in processed_ranges:
                    messages.warning(request, f"Ignored duplicate range in input: start={start}, end={end}")
                    continue
                processed_ranges.add(range_tuple)

                # Ensure indices don't exceed text length
                if end > len(source_text_content):
                    messages.warning(request, f"Ignored range out of text bounds: start={start}, end={end}")
                    continue

                # Extract the corresponding text
                selected_text = source_text_content[start:end]

                # Prepare objects for bulk insertion
                sentences_to_create.append(
                    RelevantSentence(
                        experiment=experiment,
                        text=selected_text,
                        start_char=start,
                        end_char=end
                    )
                )

            # 3. Save all new annotations to the database in a single call
            if sentences_to_create:
                RelevantSentence.objects.bulk_create(sentences_to_create)
                messages.success(request, f"{len(sentences_to_create)} highlights saved successfully.")
            else:
                messages.info(request, "No highlights to save (or all previous ones were removed).")

            # Redirect to the same page to show updated state
            return redirect(reverse('experiments:annotate_experiment', kwargs={'experiment_pk': experiment_pk}))

        except json.JSONDecodeError:
            messages.error(request, "Invalid JSON format in submitted highlights.")
        except (ValueError, TypeError) as e:
            messages.error(request, f"Error in highlight data: {e}")
        except IntegrityError as e:
            messages.error(request, f"Database error while saving: {e}")
        except Exception as e:
            messages.error(request, f"Unexpected error: {e}")

        # If any error occurred in the POST, reload the page
        # JS on the frontend will manage state persistence
        return redirect(reverse('experiments:annotate_experiment', kwargs={'experiment_pk': experiment_pk}))

    else:  # GET request
        # --- Logic to display the page ---
        # Retrieve existing annotations to pass to the template
        relevant_sentences = experiment.relevant_sentences.all().order_by('start_char')

        # Convert to simple JSON format [{start: S, end: E}, ...] for JS usage
        existing_highlights_data = [
            {"start": sent.start_char, "end": sent.end_char}
            for sent in relevant_sentences
        ]

        context = {
            'experiment': experiment,
            'source_text_content': source_text_content,
            # Pass JSON data to pre-load highlights with JS
            'existing_highlights_json': json.dumps(existing_highlights_data),
        }
        return render(request, 'experiments/annotate_experiment.html', context)
