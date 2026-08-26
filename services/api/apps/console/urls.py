from django.urls import path
from apps.console.views import TerminalCommandView, CodeExecuteView

urlpatterns = [
    path('terminal/', TerminalCommandView.as_view(), name='console-terminal'),
    path('execute/', CodeExecuteView.as_view(), name='console-execute'),
]
