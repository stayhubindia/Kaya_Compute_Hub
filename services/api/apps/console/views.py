import os
import sys
import time
import subprocess
import tempfile
import logging

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions, status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from apps.jobs.models import Job, JobStatusChoices
from services.worker.tasks.job_tasks import execute_job

from rest_framework.authentication import SessionAuthentication

class CsrfExemptSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return  # Disable CSRF check for console executions

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class TerminalCommandView(APIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        command = request.data.get('command', '').strip()
        stdin_input = request.data.get('stdin_input', request.data.get('input', None))
        if not command:
            return Response({"error": "Command string is required."}, status=status.HTTP_400_BAD_REQUEST)

        start_time = time.time()
        try:
            cwd = os.getenv("DRIVE_STORAGE_ROOT", "/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv")
            if not os.path.exists(cwd):
                cwd = "/tmp"

            env = os.environ.copy()
            venv_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.venv/bin"))
            if os.path.exists(venv_bin):
                env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

            res = subprocess.run(
                command,
                input=stdin_input if isinstance(stdin_input, str) else None,
                shell=True,
                capture_output=True,
                text=True,
                timeout=90,
                cwd=cwd,
                env=env
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            return Response({
                "command": command,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "returncode": res.returncode,
                "execution_time_ms": elapsed_ms,
                "cwd": cwd
            }, status=status.HTTP_200_OK)
        except subprocess.TimeoutExpired:
            return Response({
                "command": command,
                "error": "Command execution timed out after 60 seconds",
                "returncode": -1
            }, status=status.HTTP_408_REQUEST_TIMEOUT)
        except Exception as e:
            logger.error(f"Terminal error: {e}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class CodeExecuteView(APIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        code = request.data.get('code', '').strip()
        mode = request.data.get('mode', 'instant')  # 'instant' (Test Script) or 'job' (Background Job)
        script_name = request.data.get('script_name', 'Custom Test Script')
        target_dir = request.data.get('target_dir', '/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv')

        if not code:
            return Response({"error": "Python code string is required."}, status=status.HTTP_400_BAD_REQUEST)

        if mode == 'job':
            # Create a tracked background compute Job
            job = Job.objects.create(
                name=f"Script: {script_name}",
                description=f"User initiated Python code execution ({script_name})",
                job_type="custom_script",
                created_by=request.user,
                payload={
                    "code": code,
                    "target_dir": target_dir,
                    "script_name": script_name
                },
                status=JobStatusChoices.QUEUED,
                progress_percentage=0,
                current_stage="queued"
            )
            try:
                execute_job.apply_async(args=[str(job.id)])
            except Exception as err:
                logger.warning(f"Worker dispatch warning for job {job.id}: {err}")

            return Response({
                "status": "queued",
                "mode": "job",
                "job_id": str(job.id),
                "name": job.name,
                "message": f"Script queued as background Compute Job #{str(job.id)[:8]}"
            }, status=status.HTTP_201_CREATED)

        # Default 'instant' Test Script execution mode
        start_time = time.time()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            cwd = target_dir if os.path.exists(target_dir) else "/tmp"
            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cwd
            )
            elapsed_sec = round(time.time() - start_time, 2)
            return Response({
                "status": "success" if res.returncode == 0 else "failed",
                "mode": "instant",
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "execution_time": f"{elapsed_sec}s",
                "script_name": script_name,
                "cwd": cwd
            }, status=status.HTTP_200_OK)
        except subprocess.TimeoutExpired:
            return Response({
                "status": "failed",
                "mode": "instant",
                "error": "Script execution timed out after 120 seconds",
                "returncode": -1
            }, status=status.HTTP_408_REQUEST_TIMEOUT)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
