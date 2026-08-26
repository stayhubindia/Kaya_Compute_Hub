from django.contrib.auth import authenticate, login, logout
from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework.exceptions import AuthenticationFailed

from apps.accounts.models import User
from apps.accounts.serializers import UserSerializer, LoginSerializer
from apps.accounts.permissions import IsAuthenticatedAdmin
from apps.audit.services import log_audit_event

class LoginRateThrottle(AnonRateThrottle):
    rate = '10/minute'

class LoginView(APIView):
    permission_classes = [permissions.AllowAny]
    throttle_classes = [LoginRateThrottle]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data['email']
        password = serializer.validated_data['password']

        user = authenticate(request, email=email, password=password)
        if not user or not user.is_active:
            log_audit_event(
                action="auth.login_failure",
                resource_type="user",
                resource_id=email,
                metadata={"reason": "invalid_credentials"},
                request=request
            )
            raise AuthenticationFailed("Invalid email or password.")

        # Perform login and session rotation
        login(request, user)
        request.session.cycle_key()

        log_audit_event(
            action="auth.login_success",
            resource_type="user",
            resource_id=str(user.id),
            actor=user,
            request=request
        )

        return Response({
            "user": UserSerializer(user).data
        }, status=status.HTTP_200_OK)


class LogoutView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        if request.user and request.user.is_authenticated:
            log_audit_event(
                action="auth.logout",
                resource_type="user",
                resource_id=str(request.user.id),
                actor=request.user,
                request=request
            )
        logout(request)
        return Response({"message": "Logged out successfully"}, status=status.HTTP_200_OK)


class CurrentUserView(APIView):
    permission_classes = [IsAuthenticatedAdmin]

    def get(self, request):
        return Response(UserSerializer(request.user).data)
