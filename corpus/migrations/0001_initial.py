# Generated by Django 5.2 on 2025-05-03 08:57

import corpus.models
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SourceText',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(help_text='Titolo univoco per identificare il testo.', max_length=255, unique=True)),
                ('file', models.FileField(default=None, help_text='Carica un file .txt.', upload_to=corpus.models.source_text_upload_path, validators=[corpus.models.validate_txt_extension])),
                ('metadata', models.JSONField(blank=True, help_text='Opzionale: metadati come fonte, autore, ecc. in formato JSON.', null=True)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Question',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('source_text', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='questions', to='corpus.sourcetext')),
            ],
            options={
                'unique_together': {('source_text', 'text')},
            },
        ),
    ]
