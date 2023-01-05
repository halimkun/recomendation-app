"""django_rekomendasi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("rekomendasi/", views.rekomendasi, name="rekomendasi"),
    path("bantuan/", views.bantuan, name="bantuan"),
    path("about/", views.about, name="about"),

    path("dataset/upload/", views.upload_dataset, name="upload_dataset"),
    path("dataset/delete/", views.delete_dataset, name="delete_dataset"),
    path("dataset/bar_data/", views.bar_data, name="bar_data"),
    path("dataset/get_rekomendasi/", views.get_rekomendasi, name="get_rekomendasi"),
    path("dataset/mass-rekomendasi/", views.mass_recomendation, name="upload_dataset"),

    path("rekomendasi/print/", views.print_rekomendasi, name="print_rekomendasi"),
    path("rekomendasi/mprint/", views.print_mrekomendasi, name="print_mrekomendasi")
]
