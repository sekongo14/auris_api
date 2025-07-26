from django.urls import path
from .views import analyze_image
from .views import YoloDetectionView
from django.views.generic import TemplateView

urlpatterns = [
    # path('analyze/', analyze_image),
    path('', TemplateView.as_view(template_name="api_home.html"), name='home'),
    path("detect/", YoloDetectionView.as_view(), name="detect"),
]
