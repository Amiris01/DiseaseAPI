from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from .models import expected_disease
from .serializers import expected_diseaseSerializers
import pickle
import json
import numpy as np
from sklearn import preprocessing
import pandas as pd
from joblib import load
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
class Expected_diseaseView(viewsets.ModelViewSet):
  queryset = expected_disease.objects.all()
  serializer_class = expected_diseaseSerializers

@csrf_exempt    
@api_view(["POST"])
def predict_disease(request):
  try:
    mdl= load("./saveModels/disease_model.joblib")
    mydata=request.data
    symptoms = mydata.get("symptoms", [])
    print(mydata)
    #Preprocess data here if required
    symptom_values = [int(symptom) for symptom in symptoms] 
    symptom_values.extend([0] * (17 - len(symptom_values)))   
    symptom_values = np.reshape(symptom_values, (1, -1))
    print(symptom_values)
    y_pred_proba = mdl.predict_proba(symptom_values)
    top3_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
    top3_proba = y_pred_proba[0][top3_indices]
    label_encoder = load('./saveModels/disease_encoder.joblib')
    top3_diseases = label_encoder.inverse_transform(top3_indices)
    response_data = {
            "top_diseases": [f"{disease}: {probability:.2%}" for disease, probability in zip(top3_diseases, top3_proba)],
        }
    return JsonResponse(response_data, safe=False)
  except ValueError as e:
    return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
  
def cxcontact(request):
  return render(request,'myform/cxform.html')

def formInfo(request):
  import pandas as pd
  df1 = pd.read_csv('./Notebooks/Symptom-severity.csv')
  model = load('./saveModels/disease_model.joblib')
  
  symptoms = []
  symptom1 = request.GET['dropdown1']
  s1 = df1[df1['Symptom'] == symptom1]['weight'].iloc[0]
  symptoms.append(s1)

  symptom2 = request.GET['dropdown2']
  s2 = df1[df1['Symptom'] == symptom2]['weight'].iloc[0]
  symptoms.append(s2)

  symptom3 = request.GET['dropdown3']
  s3 = df1[df1['Symptom'] == symptom3]['weight'].iloc[0]
  symptoms.append(s3)

  symptom4 = request.GET['dropdown4']
  s4 = df1[df1['Symptom'] == symptom4]['weight'].iloc[0]
  symptoms.append(s4)

  symptom5 = request.GET['dropdown5']
  s5 = df1[df1['Symptom'] == symptom5]['weight'].iloc[0]
  symptoms.append(s5)
  
  for _ in range(12):
    symptoms.append(0)

  print([symptoms])
  y_pred_proba = model.predict_proba([symptoms])
  top3_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
  top3_proba = y_pred_proba[0][top3_indices]
  label_encoder = load('./saveModels/disease_encoder.joblib')
  top3_diseases = label_encoder.inverse_transform(top3_indices)
  result = [f"{disease}: {probability:.2%}\n" for disease, probability in zip(top3_diseases, top3_proba)]
    
  return render(request,'result.html', {'result' : result})