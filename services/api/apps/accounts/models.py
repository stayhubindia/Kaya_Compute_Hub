import uuid
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.core.exceptions import ValidationError

class UserManager(BaseUserManager):
    def create_admin(self, email, password=None):
        if not email:
            raise ValueError('The Email field must be set')
        
        # Enforce single active admin rule
        if self.model.objects.filter(is_active=True).exists():
            raise ValidationError('An active admin account already exists. Only one active admin account is allowed.')

        email = self.normalize_email(email)
        user = self.model(email=email, is_active=True)
        if password:
            user.set_password(password)
        else:
            raise ValueError('A password must be provided for the admin account.')
        
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        # Fallback for create_user to use create_admin logic
        return self.create_admin(email, password)

    def create_superuser(self, email, password=None, **extra_fields):
        return self.create_admin(email, password)


class User(AbstractBaseUser):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True, db_index=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    class Meta:
        db_table = 'accounts_user'
        ordering = ['-created_at']

    def __str__(self):
        return f"Admin ({self.email})"
