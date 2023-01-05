from django import forms
from django.core.validators import FileExtensionValidator


class UploadFileForm(forms.Form):
    dataset = forms.FileField(
        label='Select a file',
        allow_empty_file=False,
        validators=[FileExtensionValidator(allowed_extensions=['csv', 'xlsx', 'xls'])],
    )


def handle_upload_dataset(f):
    with open('media/dset/' + f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def handle_upload_dtest(f):
    with open('media/dtest/' + f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)