from flask import Flask, render_template, jsonify, request
import pickle
import pandas as pd
import numpy as np

# creamos objeto flask
app = Flask(__name__) # __name__ es alias de nombre del archivo

# se carga el modelo
loaded_model = pickle.load(open('models/coffee_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

# en flask app.route se debe escribir toda la palabra string

#prediction function
def ValuePredictor(to_predict_list):
    cols = ['country_of_origin', 'variety', 'aroma', 'aftertaste', 'acidity', 'body', 'balance', 'moisture']
    to_predict = pd.DataFrame(np.array(to_predict_list).reshape(1,8), columns = cols)
    result = loaded_model.predict(to_predict)
    return result.tolist()[0]


@app.route('/result/', methods = ('GET', 'POST'))
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        #to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if result == 'Yes':
            prediction = jsonify(message = 'Yes it is a specialty coffee')
        else:
            prediction = jsonify(message = 'It is not a specialty coffee')
            
        return render_template("result.html", prediction)

#@app.route('/<string:country>/<string:variety>/<float:aroma>/<float:aftertaste>/<float:acidity>/<float:body>/<float:balance>/<float:moisture>')
#def result(country, variety, aroma, aftertaste, acidity, body, balance, moisture):
#    cols = ['country_of_origin', 'variety', 'aroma', 'aftertaste', 'acidity', 'body', 'balance', 'moisture']
#    data = [country, variety, aroma, aftertaste, acidity, body, balance, moisture]
#    posted = pd.DataFrame(np.array(data).reshape(1,8), columns = cols)
#    # se predice con datos creados
#    result = loaded_model.predict(posted)
#    # salida a mostrar
#    text_result = result.tolist()[0]
#    if text_result == 'Yes':
#        return jsonify(message = 'Si es un cafe de especialidad'), 200 # 200 es el código de error
#    else:
#        return jsonify(message = 'No es un cafe de especialidad'), 200

if __name__ == '__main__':
    app.run(debug = True, host = '127.0.0.1', port = 5000)

## prueba para si: Guatemala/Bourbon/7.83/7.67/7.33/7.67/7.67/0.11
# ## prueba para no: Other/Other/7.25/6.83/7.17/7.00/7.17/0.11

#### ejecutar en terminal en dos líneas distintas:
#export FLASK_APP=src/app.py
#export FLASK_ENV=development
#run flask

### https://nightlycommit.github.io/twing/language-reference/tags/extends.html