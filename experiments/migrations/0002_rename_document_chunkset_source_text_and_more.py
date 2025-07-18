# Generated by Django 5.2 on 2025-05-03 20:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('corpus', '0001_initial'),
        ('experiments', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='chunkset',
            old_name='document',
            new_name='source_text',
        ),
        migrations.RenameField(
            model_name='experiment',
            old_name='document',
            new_name='source_text',
        ),
        migrations.AlterUniqueTogether(
            name='chunkset',
            unique_together={('source_text', 'strategy')},
        ),
        migrations.AlterUniqueTogether(
            name='experiment',
            unique_together={('source_text', 'question')},
        ),
    ]
