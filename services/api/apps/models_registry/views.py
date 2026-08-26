from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.permissions import IsAuthenticatedAdmin
from apps.models_registry.models import ModelVersion, ModelStatusChoices
from apps.models_registry.serializers import ModelVersionSerializer, ModelVersionCreateSerializer
from apps.training.models import TrainingRun, TrainingRunStatus
from apps.audit.services import log_audit_event

class ModelVersionViewSet(viewsets.ModelViewSet):
    serializer_class = ModelVersionSerializer
    permission_classes = [IsAuthenticatedAdmin]

    def get_queryset(self):
        return ModelVersion.objects.all()

    def create(self, request, *args, **kwargs):
        serializer = ModelVersionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        name = serializer.validated_data['name']
        version = serializer.validated_data['version']

        if ModelVersion.objects.filter(name=name, version=version).exists():
            return Response(
                {"error": {"code": "DUPLICATE_MODEL_VERSION", "message": f"Model version {name}:{version} already exists."}},
                status=status.HTTP_400_BAD_REQUEST
            )

        training_run_id = serializer.validated_data.get('training_run_id')
        training_run = None
        if training_run_id:
            try:
                training_run = TrainingRun.objects.get(id=training_run_id)
            except TrainingRun.DoesNotExist:
                return Response({"error": {"code": "TRAINING_RUN_NOT_FOUND", "message": "Associated training run not found."}}, status=status.HTTP_404_NOT_FOUND)

        model_version = ModelVersion.objects.create(
            name=name,
            version=version,
            training_run=training_run,
            framework=serializer.validated_data['framework'],
            framework_version=serializer.validated_data.get('framework_version', ''),
            model_format=serializer.validated_data['model_format'],
            checksum=serializer.validated_data['checksum'],
            metadata=serializer.validated_data.get('metadata', {}),
            status=ModelStatusChoices.REGISTERED,
            created_by=request.user
        )

        log_audit_event(
            action="model.registered",
            resource_type="model_version",
            resource_id=str(model_version.id),
            actor=request.user,
            metadata={"name": name, "version": version, "checksum": model_version.checksum},
            request=request
        )

        return Response(ModelVersionSerializer(model_version).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticatedAdmin])
    def approve(self, request, pk=None):
        model_ver = self.get_object()

        # Approval Rules Validation
        if not model_ver.checksum:
            return Response({"message": "Artifact checksum is missing."}, status=status.HTTP_400_BAD_REQUEST)

        if model_ver.training_run and model_ver.training_run.status != TrainingRunStatus.SUCCEEDED:
            return Response({"message": "Cannot approve model from unsuccessful training run."}, status=status.HTTP_400_BAD_REQUEST)

        model_ver.status = ModelStatusChoices.APPROVED
        model_ver.save(update_fields=['status'])

        log_audit_event(
            action="model.approved",
            resource_type="model_version",
            resource_id=str(model_ver.id),
            actor=request.user,
            metadata={"name": model_ver.name, "version": model_ver.version},
            request=request
        )

        return Response({"id": str(model_ver.id), "status": model_ver.status, "message": "Model version approved successfully."})

    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticatedAdmin])
    def archive(self, request, pk=None):
        model_ver = self.get_object()
        model_ver.status = ModelStatusChoices.ARCHIVED
        model_ver.save(update_fields=['status'])

        log_audit_event(
            action="model.archived",
            resource_type="model_version",
            resource_id=str(model_ver.id),
            actor=request.user,
            metadata={"name": model_ver.name, "version": model_ver.version},
            request=request
        )

        return Response({"id": str(model_ver.id), "status": model_ver.status, "message": "Model version archived."})
