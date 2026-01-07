from django.shortcuts import render
from .utils import (
    get_dataset_info,
    get_visualization_data,
    get_division_data,
    get_preparation_data,
    get_evaluation_data,
    check_dataset_exists,
)


def introduccion(request):
    dataset_exists = check_dataset_exists()
    context = {
        'dataset_exists': dataset_exists,
    }
    if dataset_exists:
        info, _ = get_dataset_info()
        context['info'] = info
    return render(request, 'dashboard/introduccion.html', context)


def visualizacion(request):
    dataset_exists = check_dataset_exists()
    context = {'dataset_exists': dataset_exists}
    
    if dataset_exists:
        try:
            data = get_visualization_data()
            context.update(data)
        except Exception as e:
            context['error'] = str(e)
    
    return render(request, 'dashboard/visualizacion.html', context)


def division(request):
    dataset_exists = check_dataset_exists()
    context = {'dataset_exists': dataset_exists}
    
    if dataset_exists:
        try:
            data = get_division_data()
            context.update(data)
        except Exception as e:
            context['error'] = str(e)
    
    return render(request, 'dashboard/division.html', context)


def preparacion(request):
    dataset_exists = check_dataset_exists()
    context = {'dataset_exists': dataset_exists}
    
    if dataset_exists:
        try:
            data = get_preparation_data()
            context.update(data)
        except Exception as e:
            context['error'] = str(e)
    
    return render(request, 'dashboard/preparacion.html', context)


def pipelines(request):
    dataset_exists = check_dataset_exists()
    context = {'dataset_exists': dataset_exists}
    
    if dataset_exists:
        try:
            data = get_preparation_data()
            context.update(data)
        except Exception as e:
            context['error'] = str(e)
    
    return render(request, 'dashboard/pipelines.html', context)


def evaluacion(request):
    dataset_exists = check_dataset_exists()
    context = {'dataset_exists': dataset_exists}
    
    if dataset_exists:
        try:
            data = get_evaluation_data()
            context.update(data)
        except Exception as e:
            context['error'] = str(e)
    
    return render(request, 'dashboard/evaluacion.html', context)
