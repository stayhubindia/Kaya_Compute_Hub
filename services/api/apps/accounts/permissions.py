from rest_framework import permissions

class IsAuthenticatedAdmin(permissions.BasePermission):
    """
    Allows access only to the authenticated active single admin.
    """
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.is_active)

    def has_object_permission(self, request, view, obj):
        return self.has_permission(request, view)

# Backward-compatibility permission aliases pointing to IsAuthenticatedAdmin
IsAdmin = IsAuthenticatedAdmin
IsOperatorOrAdmin = IsAuthenticatedAdmin
IsOwnerOrOperator = IsAuthenticatedAdmin
IsAuthenticatedAndActive = IsAuthenticatedAdmin
IsAdminOrOperator = IsAuthenticatedAdmin
IsOwnerOrAdminOrOperator = IsAuthenticatedAdmin
