import pickle
import pandas as pd
import numpy as np

## Para resultado Yes
"""country = 'Colombia'
variety = 'Caturra'
aroma = 7.83
aftertaste = 7.67
acidity = 7.33
body = 7.67
balance = 7.67
moisture = 0.11"""


## Para resultado No
"""country = 'Other'
variety = 'Other'
aroma = 7.25
aftertaste = 6.83
acidity = 7.17
body = 7.00
balance = 7.17
moisture = 0.11"""

# datos para probar el modelo
cols = ['country_of_origin', 'variety', 'aroma', 'aftertaste', 'acidity', 'body', 'balance', 'moisture']
data = [country, variety, aroma, aftertaste, acidity, body, balance, moisture]
posted = pd.DataFrame(np.array(data).reshape(1,8), columns = cols)

# se carga el modelo
loaded_model = pickle.load(open('models/coffee_model.pkl', 'rb'))

# se predice con datos creados
result = loaded_model.predict(posted)

# salida a mostrar
text_result = result.tolist()[0]
print(text_result)