from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib import messages
from django.db import IntegrityError, transaction
from .models import SourceText, Question
from .forms import SourceTextForm, QuestionForm
from corpus.service import source_text_service
from experiments.models import Experiment
import json


def list_source_texts(request):
    """Displays the list of all uploaded source texts."""
    source_texts = SourceText.objects.all().order_by('-uploaded_at')
    context = {'source_texts': source_texts}
    return render(request, 'corpus/list_source_texts.html', context)


def upload_source_text(request):
    """Handles the upload of a new source text file."""
    if request.method == 'POST':
        form = SourceTextForm(request.POST, request.FILES)
        if form.is_valid():
            metadata_str = form.cleaned_data.get('metadata')
            metadata_dict = None
            if metadata_str:
                try:
                    # Load validated string as Python dict
                    metadata_dict = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                except json.JSONDecodeError:
                    # Should not happen if clean_metadata works correctly
                    messages.error(request, "Internal error while decoding metadata.")
                    return render(request, 'corpus/upload_source_text.html', {'form': form})

            instance = form.save(commit=False)
            instance.metadata = metadata_dict
            instance.save()

            messages.success(request, f"Source text '{instance.title}' uploaded successfully.")
            return redirect(reverse('corpus:list_source_texts'))
        else:
            messages.error(request, "Upload failed. Please check the form for errors.")
    else:
        form = SourceTextForm()

    context = {'form': form}
    return render(request, 'corpus/upload_source_text.html', context)


@transaction.atomic
def source_text_detail(request, pk):
    """
    Displays the details of a source text, allows adding new questions,
    and automatically creates an associated experiment.
    """
    source_text = get_object_or_404(SourceText, pk=pk)
    questions = source_text.questions.all()  # Ordering handled in model Meta

    if request.method == 'POST':
        question_form = QuestionForm(request.POST)
        if question_form.is_valid():
            try:
                new_question = question_form.save(commit=False)
                new_question.source_text = source_text
                new_question.save()

                # --- AUTOMATIC EXPERIMENT CREATION ---
                experiment, created = Experiment.objects.get_or_create(
                    source_text=source_text,
                    question=new_question,
                    # Optionally: defaults={'description': 'Automatically created'}
                )
                if created:
                    messages.info(request, f"New experiment (ID: {experiment.id}) automatically created for this question.")
                # --- END AUTOMATIC CREATION ---

                messages.success(request, "Question added successfully.")
                return redirect(reverse('corpus:source_text_detail', kwargs={'pk': pk}))

            except IntegrityError:
                messages.error(request, "This question already exists for this text.")
            except Exception as e:
                messages.error(request, f"Error while adding question or creating experiment: {e}")
        else:
            messages.error(request, "Unable to add question. Please check the form.")
    else:
        question_form = QuestionForm()

    context = {
        'source_text': source_text,
        'source_text_content': source_text_service.get_full_text(source_text),
        'questions': questions,
        'question_form': question_form,
    }
    return render(request, 'corpus/source_text_detail.html', context)
