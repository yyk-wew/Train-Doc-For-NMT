#!/usr/bin/python
# -*- coding: UTF-8 -*-

from django.shortcuts import render
from django.shortcuts import HttpResponse
from . import model

def index(request):
    context = {}
    if request.method == "POST" and request.POST['code1']:
        if "submit" in request.POST:
            code1 = request.POST['code1']
            code2 = model.translate(code1,0)
            context["answer"] = "English"
            context["code1"] = code1
            context["code2"] = code2

        elif "reset" in request.POST:
            context["code1"] = ''
            context["code2"] = ''
            context["answer"] = ''

    return render(request,'check.html',context)

