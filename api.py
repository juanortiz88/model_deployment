#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
from flask_restx import Api, Resource, fields
from sklearn.externals import joblib


# In[ ]:


app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title = 'Prediciendo el precio de un automóvil',
    description = 'Modelo XGBoost para predecir el precio')

ns = api.namespace('predict', 
     description='Auto price predictor')
   
parser = api.parser()

parser.add_argument(
    'Year/Año', 
    type=int, 
    required=True, 
    help='Año/Modelo del vehículo', 
    location='args')

parser.add_argument(
    'Mileage/Millaje', 
    type=int, 
    required=True, 
    help='Millas recorridas en el vehículo', 
    location='args')

parser.add_argument(
    'State/Estado', 
    type=str, 
    required=True, 
    help='Estado de los Estados Unidos de América. Ingresar un argumento de máximo 2 posiciones', 
    location='args')

parser.add_argument(
    'Make/Fabricante', 
    type=str, 
    required=True, 
    help='Fabricante del vehículo', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Modelo del vehículo', 
    location='args')


resource_fields = api.model('Resource', {
    'result': fields.String,
})


# In[ ]:


from model_deployment.model_deployment_price import predict_price


# In[ ]:


@ns.route('/')

class PriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        years = args['Year/Año'] 
        mil = args['Mileage/Millaje'] 
        stat = args['State/Estado']
        meik = args['Make/Fabricante']
        mod = args['Model']
        return {
         "result": predict_price(years, mil, stat, meik, mod)
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


# In[ ]:




