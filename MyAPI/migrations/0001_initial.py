# Generated by Django 5.0.4 on 2024-04-16 15:13

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='expected_disease',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symptom1', models.CharField(max_length=50)),
                ('symptom2', models.CharField(max_length=50)),
                ('symptom3', models.CharField(max_length=50)),
                ('symptom4', models.CharField(max_length=50)),
                ('symptom5', models.CharField(max_length=50)),
                ('symptom6', models.CharField(max_length=50)),
                ('symptom7', models.CharField(max_length=50)),
                ('symptom8', models.CharField(max_length=50)),
                ('symptom9', models.CharField(max_length=50)),
                ('symptom10', models.CharField(max_length=50)),
                ('symptom11', models.CharField(max_length=50)),
                ('symptom12', models.CharField(max_length=50)),
                ('symptom13', models.CharField(max_length=50)),
                ('symptom14', models.CharField(max_length=50)),
                ('symptom15', models.CharField(max_length=50)),
                ('symptom16', models.CharField(max_length=50)),
                ('symptom17', models.CharField(max_length=50)),
            ],
        ),
    ]
