from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('integrations', '0002_externalrun_selected_google_account_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='connectedaccount',
            name='encrypted_credential_json',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.DeleteModel(
            name='OAuthState',
        ),
    ]
