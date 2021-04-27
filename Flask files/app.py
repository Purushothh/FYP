from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# loaded models for prediction
aaplModelTdy = pickle.load(
    open('D:\\Applications\\Project Folders\\StockPrediction\\models\\aapl_tdyPrediction.pkl', 'rb'))
aaplModelTmrw = pickle.load(
    open('D:\\Applications\\Project Folders\\StockPrediction\\models\\aapl_tmrwPrediction.pkl', 'rb'))
dwjaModelTdy = pickle.load(
    open('D:\\Applications\\Project Folders\\StockPrediction\\models\\DWJA_tdysPrediction.pkl', 'rb'))
dwjaModelTmrw = pickle.load(
    open('D:\\Applications\\Project Folders\\StockPrediction\\models\\DWJA_tmrwPrediction.pkl', 'rb'))
nlpModel = pickle.load(open('D:\\Applications\\Project Folders\\StockPrediction\\models\\nlp_model.pkl', 'rb'))
transform_model = pickle.load(
    open('D:\\Applications\\Project Folders\\StockPrediction\\models\\countvector_model.pkl', 'rb'))


# routing of pages
@app.route('/')
def hello_world():
    return render_template('home.html')


# to apple page
@app.route('/aapl')
def hello_world1():
    return render_template('AAPLPredict.html')


# to dow jones page
@app.route('/dwja')
def hello_world2():
    return render_template('DWJAPredict.html')


# function to predict the price for AAPL and stock and news
@app.route('/predictAAPL', methods=['POST'])
def predict():
    # get stock values from the front end.
    getVal1 = float(request.form['ph1'])
    getVal2 = float(request.form['ph2'])
    getVal3 = float(request.form['ph3'])
    getVal4 = float(request.form['ph4'])
    finalarray = np.array([[getVal1, getVal2, getVal3, getVal4]])  # convert into an array

    # get news topic from the front end.
    getNewsVal = str(request.form['ph'])
    news_array = [getNewsVal]  # convert into an array
    predictions = nlpModel.predict(transform_model.transform(news_array))  # prediction
    pred = predictions[0]

    # condition to suggestion for trade
    if pred == 0:
        predictText = 'Not a good day to trade'
    else:
        predictText = 'Good day to trade'

    # prediction for the two days with .2 decimal places
    prediction = '%.2f' % aaplModelTdy.predict(finalarray)
    tomoPrediction = '%.2f' % aaplModelTmrw.predict(finalarray)

    # return all values to the front end.
    return render_template('AAPLPredict.html', data=predictText, data1=prediction, data2=tomoPrediction)


# function to predict the price for AAPL and stock and news
@app.route('/predictDWJA', methods=['POST'])
def predict2():
    # get stock values from the front end.
    getVal1 = float(request.form['ph1'])
    getVal2 = float(request.form['ph2'])
    getVal3 = float(request.form['ph3'])
    getVal4 = float(request.form['ph4'])
    finalarray = np.array([[getVal1, getVal2, getVal3, getVal4]])

    # prediction for the two days with .2 decimal places
    predictionDWJA = '%.2f' % dwjaModelTdy.predict(finalarray)
    predictionDWJAtomo = '%.2f' % dwjaModelTmrw.predict(finalarray)

    # get news topic from the front end.
    getNewsVal = str(request.form['ph'])
    news_array = [getNewsVal]
    predictions = nlpModel.predict(transform_model.transform(news_array))
    pred = predictions[0]

    # condition to suggestion for trade
    if pred == 0:
        predictText = 'Not a good day to trade'
    else:
        predictText = 'Good day to trade'

    # return all values to the front end.
    return render_template('DWJAPredict.html', data=predictionDWJA, data1=predictionDWJAtomo, data2=predictText)


if __name__ == '__main__':
    app.run()
