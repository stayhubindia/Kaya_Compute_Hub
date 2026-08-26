import getpass
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from apps.accounts.models import User

class Command(BaseCommand):
    help = 'Create the single private admin account for Kaya Compute Hub.'

    def add_arguments(self, parser):
        parser.add_argument('--email', type=str, help='Admin email address')
        parser.add_argument('--password', type=str, help='Admin password')

    def handle(self, *args, **options):
        # 1. Enforce single active admin rule
        if User.objects.filter(is_active=True).exists():
            raise CommandError('An active admin account already exists. Only one active admin account is permitted.')

        self.stdout.write(self.style.NOTICE('=== Kaya Compute Hub - Single Admin Creation ==='))

        email = options.get('email')
        password = options.get('password')

        # 2. Prompt for Email if not provided via options
        if not email:
            email = input('Enter Admin Email: ').strip()
        if not email:
            raise CommandError('Email address cannot be empty.')

        # 3. Prompt for Password if not provided via options
        if not password:
            password = getpass.getpass('Enter Admin Password: ')
            confirm_password = getpass.getpass('Confirm Admin Password: ')

            if password != confirm_password:
                raise CommandError('Passwords do not match.')

        # 4. Reject weak passwords using Django password validators
        try:
            validate_password(password)
        except ValidationError as err:
            raise CommandError(f'Password strength validation failed: {"; ".join(err.messages)}')

        # 5. Create Admin Account (saves only password hash)
        try:
            admin_user = User.objects.create_admin(email=email, password=password)
            self.stdout.write(self.style.SUCCESS(f'Successfully created single admin account for {admin_user.email}.'))
        except Exception as e:
            raise CommandError(f'Failed to create admin account: {str(e)}')
