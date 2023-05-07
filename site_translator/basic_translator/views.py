from django.http import HttpResponse, HttpResponseNotFound, Http404
from django.shortcuts import render
from django_ajax.decorators import ajax
from .forms import InputForm, SelectForm_to, SelectForm_from
from .utils import translate

def index(request):
    input_form = InputForm()
    select_form_to = SelectForm_to()
    select_form_from = SelectForm_from()
    return render(request, 'index.html', {'title': 'Переводчик','input_form': input_form, 'select_form_to': select_form_to, 'select_form_from': select_form_from})

@ajax
def ajax_request(request):
    input_data = request.POST.get('input_data')
    select_data_to = request.POST.get('select_data_to')
    select_data_from = request.POST.get('select_data_from')
    output_data = translate(input_data, select_data_from, select_data_to) 
    print(f'\n{output_data}\n')
    return {'output_data': output_data}

def pageNotFound(request, exception):
    return HttpResponseNotFound('<h1>Страница не найдена</h1>')