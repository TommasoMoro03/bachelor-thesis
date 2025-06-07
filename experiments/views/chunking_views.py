import json
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib import messages
from django.db import transaction, IntegrityError
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator

from experiments.models import Experiment, RelevantSentence, ChunkingStrategy, ChunkSet, Chunk
from experiments.forms import ChunkingStrategyForm
from corpus.models import SourceText
from corpus.service import source_text_service
from experiments.service import chunk_implementations


# --- ChunkingStrategy Views (CRUD) ---

def list_strategies(request):
    """Display a list of all defined chunking strategies."""
    strategies = ChunkingStrategy.objects.all().order_by('method_type')
    context = {'strategies': strategies}
    return render(request, 'experiments/list_strategies.html', context)

def create_strategy(request):
    """Create a new chunking strategy."""
    if request.method == 'POST':
        form = ChunkingStrategyForm(request.POST)
        if form.is_valid():
            try:
                form.save()
                messages.success(request, f"Strategy '{form.cleaned_data['name']}' created successfully.")
                return redirect(reverse('experiments:list_strategies'))
            except IntegrityError:
                form.add_error('name', "A strategy with this name already exists.")
                messages.error(request, "Error: Duplicate strategy name.")
            except Exception as e:
                messages.error(request, f"Unexpected error: {e}")
                form.add_error(None, f"Unexpected error: {e}")
        else:
            messages.error(request, "Error creating the strategy. Please check the form.")
    else:
        form = ChunkingStrategyForm()

    context = {'form': form, 'is_edit': False}
    return render(request, 'experiments/strategy_form.html', context)

def edit_strategy(request, pk):
    """Edit an existing chunking strategy."""
    strategy = get_object_or_404(ChunkingStrategy, pk=pk)
    if request.method == 'POST':
        form = ChunkingStrategyForm(request.POST, instance=strategy)
        if form.is_valid():
            try:
                form.save()
                messages.success(request, f"Strategy '{strategy.name}' updated successfully.")
                return redirect(reverse('experiments:list_strategies'))
            except IntegrityError:
                form.add_error('name', "A strategy with this name already exists.")
                messages.error(request, "Error: Duplicate strategy name.")
            except Exception as e:
                messages.error(request, f"Unexpected error: {e}")
                form.add_error(None, f"Unexpected error: {e}")
        else:
            messages.error(request, "Error updating the strategy. Please check the form.")
    else:
        # Pre-fill form with formatted JSON
        initial_params = json.dumps(strategy.parameters, indent=2) if isinstance(strategy.parameters, dict) else strategy.parameters
        form = ChunkingStrategyForm(instance=strategy, initial={'parameters': initial_params})

    context = {'form': form, 'strategy': strategy, 'is_edit': True}
    return render(request, 'experiments/strategy_form.html', context)

@require_POST
def delete_strategy(request, pk):
    """Delete a chunking strategy (POST only)."""
    strategy = get_object_or_404(ChunkingStrategy, pk=pk)
    try:
        strategy_name = strategy.name
        if ChunkSet.objects.filter(strategy=strategy).exists():
            messages.error(request, f"Cannot delete strategy '{strategy_name}' because it is in use.")
            return redirect(reverse('experiments:list_strategies'))

        strategy.delete()
        messages.success(request, f"Strategy '{strategy_name}' deleted successfully.")
    except Exception as e:
        messages.error(request, f"Error deleting strategy: {e}")

    return redirect(reverse('experiments:list_strategies'))


# --- Chunk Management Views ---

def manage_document_chunking(request, source_text_pk):
    """View chunking status of a document and allow applying strategies."""
    source_text = get_object_or_404(SourceText, pk=source_text_pk)
    available_strategies = ChunkingStrategy.objects.all().order_by('name')
    existing_chunk_sets = ChunkSet.objects.filter(source_text=source_text)

    applied_chunk_sets_map = {cs.strategy_id: cs.pk for cs in existing_chunk_sets}

    context = {
        'source_text': source_text,
        'available_strategies': available_strategies,
        'applied_chunk_sets_map': applied_chunk_sets_map,
    }
    return render(request, 'experiments/manage_document_chunking.html', context)

@require_POST
@transaction.atomic
def apply_strategy_to_document(request, source_text_pk, strategy_pk):
    """Apply a strategy to a document, creating a ChunkSet and its Chunks."""
    source_text = get_object_or_404(SourceText, pk=source_text_pk)
    strategy = get_object_or_404(ChunkingStrategy, pk=strategy_pk)

    if ChunkSet.objects.filter(source_text=source_text, strategy=strategy).exists():
        messages.warning(request, f"Chunks for '{source_text.title}' using strategy '{strategy.name}' already exist.")
        return redirect(reverse('experiments:manage_document_chunking', kwargs={'source_text_pk': source_text_pk}))

    content = source_text_service.get_full_text(source_text)

    try:
        chunks_data = chunk_implementations.apply_chunking_strategy(strategy, content)
        chunk_set = ChunkSet.objects.create(source_text=source_text, strategy=strategy)

        chunks_to_create = [
            Chunk(
                chunk_set=chunk_set,
                text=data['text'],
                chunk_index=i,
                start_char=data['start_char'],
                end_char=data['end_char']
            )
            for i, data in enumerate(chunks_data)
        ]

        if chunks_to_create:
            Chunk.objects.bulk_create(chunks_to_create)
            messages.success(request, f"{len(chunks_to_create)} chunks created for '{source_text.title}' using '{strategy.name}'.")
        else:
            messages.warning(request, f"No chunks generated for '{source_text.title}' with strategy '{strategy.name}'.")

    except ValueError:
        messages.error(request, f"Unsupported method type '{strategy.method_type}'.")
    except Exception as e:
        messages.error(request, f"Error applying strategy '{strategy.name}': {e}")

    return redirect(reverse('experiments:manage_document_chunking', kwargs={'source_text_pk': source_text_pk}))

def view_chunk_set(request, chunk_set_pk):
    """View chunks and highlight relevant sentences (optional via GET experiment_id)."""
    chunk_set = get_object_or_404(
        ChunkSet.objects.select_related('source_text', 'strategy'),
        pk=chunk_set_pk
    )
    chunk_list = chunk_set.chunks.all()

    experiment_id = request.GET.get('experiment_id')
    existing_highlights_data = []
    experiment = None

    if experiment_id:
        try:
            experiment = get_object_or_404(
                Experiment.objects.select_related('question'),
                pk=int(experiment_id),
                source_text=chunk_set.source_text
            )
            relevant_sentences = experiment.relevant_sentences.all()
            existing_highlights_data = [
                {"start": sent.start_char, "end": sent.end_char}
                for sent in relevant_sentences
            ]
        except (ValueError, TypeError):
            messages.warning(request, "Invalid experiment ID.")
        except Experiment.DoesNotExist:
            messages.warning(request, f"No experiment found with ID {experiment_id}")
            experiment = None

    paginator = Paginator(chunk_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'chunk_set': chunk_set,
        'page_obj': page_obj,
        'experiment': experiment,
        'existing_highlights_json': json.dumps(existing_highlights_data),
    }
    return render(request, 'experiments/view_chunk_set.html', context)

@require_POST
def delete_chunk_set(request, chunk_set_pk):
    """Delete a ChunkSet and all associated Chunks (POST only)."""
    chunk_set = get_object_or_404(ChunkSet, pk=chunk_set_pk)
    source_text_pk = chunk_set.source_text.pk
    strategy_name = chunk_set.strategy.name

    try:
        with transaction.atomic():
            chunk_set.delete()
            messages.success(request, f"Chunk set for strategy '{strategy_name}' deleted successfully.")
    except Exception as e:
        messages.error(request, f"Error deleting chunk set for strategy '{strategy_name}': {e}")

    return redirect(reverse('experiments:manage_document_chunking', kwargs={'source_text_pk': source_text_pk}))
