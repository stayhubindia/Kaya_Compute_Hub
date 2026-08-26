from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status

def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is not None:
        if isinstance(response.data, dict):
            detail = response.data.get('detail')
            if detail:
                message = str(detail)
                details = response.data
            else:
                message = "Validation or processing error."
                details = response.data
        elif isinstance(response.data, list):
            message = "Validation errors occurred."
            details = response.data
        else:
            message = str(response.data)
            details = response.data

        response.data = {
            "error": {
                "status_code": response.status_code,
                "message": message,
                "details": details
            }
        }
    return response
