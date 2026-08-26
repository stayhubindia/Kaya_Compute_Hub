from rest_framework import serializers
from apps.accounts.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'is_active', 'last_login', 'created_at', 'updated_at']
        read_only_fields = ['id', 'is_active', 'last_login', 'created_at', 'updated_at']

class LoginSerializer(serializers.Serializer):
    email = serializers.CharField()
    password = serializers.CharField(write_only=True)
