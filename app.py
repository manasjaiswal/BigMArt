from flask import Flask,render_template,url_for,app,request,jsonify
import pandas as pd
import numpy as np
import pickle

regressor=pickle.load(open('bigmart_pred','rb'))
li=[{'DR': 1496.7184, 'FD': 1810.976, 'NC': 1874.8928},
 {'LF': 1765.0358, 'reg': 1844.5989},
 {'Baking Goods': 1577.946,
  'Breads': 1860.2452,
  'Breakfast': 1554.643,
  'Canned': 1860.2452,
  'Dairy': 1650.8510999999999,
  'Frozen Foods': 1687.1372,
  'Fruits and Vegetables': 1830.95,
  'Hard Drinks': 1816.6353,
  'Health and Hygiene': 1669.4935,
  'Household': 1981.4208,
  'Meat': 1829.6184,
  'Others': 1713.7692,
  'Seafood': 2055.3246,
  'Snack Foods': 1944.136,
  'Soft Drinks': 1518.024,
  'Starchy Foods': 1968.1048},
 {'OUT010': 250.3408,
  'OUT013': 2050.664,
  'OUT017': 2005.0567,
  'OUT018': 1655.1788,
  'OUT019': 265.3213,
  'OUT027': 3364.9532,
  'OUT035': 2109.2544,
  'OUT045': 1834.9448,
  'OUT046': 1945.8004999999998,
  'OUT049': 1966.1074},
 {'High': 2050.664, 'Medium': 2251.0698, 'Small': 1484.0682},
 {'Tier 1': 1487.3971999999999, 'Tier 2': 2004.058, 'Tier 3': 1812.3076},
 {'Grocery Store': 256.9988,
  'Supermarket Type1': 1990.742,
  'Supermarket Type2': 1655.1788,
  'Supermarket Type3': 3364.9532}]

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predictions():
    data=[x for x in request.form.values()]
    print(data)
    data[3]=data[3][0:2]
    for i in range(7):
        data[i+3]=li[i][data[i+3]]
    data[10]=2013-int(data[10])
    new_data=[]
    new_data.append(float(data[1]))
    new_data.append(data[5])
    new_data.append(data[3])
    new_data.append(float(data[0]))
    new_data.append(float(data[2]))
    new_data.append(data[4])
    new_data.append(data[6])
    new_data.append(data[7])
    new_data.append(data[8])
    new_data.append(data[9])
    new_data.append(data[10])
    output=regressor.predict([new_data])[0]
    return render_template('home.html',prediction_result=f'The sale of item is Rs.{np.round(output,3)}/-')


if __name__=='__main__' :
    app.run(debug=True)
