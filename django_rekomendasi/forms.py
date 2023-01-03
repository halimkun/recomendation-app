from django import forms
from django.core.validators import FileExtensionValidator


class UploadFileForm(forms.Form):
    dataset = forms.FileField(
        label='Select a file',
    )


def handle_uploaded_file(f):
    with open('media/' + f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
