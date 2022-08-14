from flask import Flask, render_template, jsonify, request
import pickle
import pandas as pd
import numpy as np

# creamos objeto flask
app = Flask(__name__) # __name__ es alias de nombre del archivo

# se carga el modelo
loaded_model = pickle.load(open('models/coffee_model.pkl', 'rb'))

@app.route('/')
def show_home():
    return render_template('index.html')

# en flask app.route se debe escribir toda la palabra string
@app.route('/<string:country>/<string:variety>/<float:aroma>/<float:aftertaste>/<float:acidity>/<float:body>/<float:balance>/<float:moisture>')
def result(country, variety, aroma, aftertaste, acidity, body, balance, moisture):
    cols = ['country_of_origin', 'variety', 'aroma', 'aftertaste', 'acidity', 'body', 'balance', 'moisture']
    data = [country, variety, aroma, aftertaste, acidity, body, balance, moisture]
    posted = pd.DataFrame(np.array(data).reshape(1,8), columns = cols)
    # se predice con datos creados
    result = loaded_model.predict(posted)
    # salida a mostrar
    text_result = result.tolist()[0]
    if text_result == 'Yes':
        return jsonify(message = 'Si es un cafe de especialidad'), 200 # 200 es el código de error
    else:
        return jsonify(message = 'No es un cafe de especialidad'), 200

if __name__ == '__main__':
    app.run(debug = True, host = '127.0.0.1', port = 5000)

## prueba para si: Guatemala/Bourbon/7.83/7.67/7.33/7.67/7.67/0.11
# ## prueba para no: Other/Other/7.25/6.83/7.17/7.00/7.17/0.11

#### ejecutar en terminal en dos líneas distintas:
#export FLASK_APP=src/app.py
#run flask