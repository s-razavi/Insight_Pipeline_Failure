from flask import Flask, render_template, request
import pandas as pd
import re
from nltk.tokenize import word_tokenize

app = Flask(__name__)

def outputs(some_input):
    import numpy as np
    import pickle
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.optimizers import SGD
    from keras.models import model_from_json
    
    vector = pickle.load(open('vector.pkl','rb'))
    encoded_y = pickle.load(open('encoded_y.pkl','rb'))
    mileage_df = pickle.load(open('mileage.pkl','rb'))
    words = pickle.load(open('used_words.pkl','rb'))
    x, x_test_final, y, y_test_final = train_test_split(vector, encoded_y, test_size=0.1, random_state=44)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,  random_state=44)
    TFIDF = pickle.load(open('TFIDF.pkl','rb'))
    label_encoder_y = pickle.load(open('label_encoder_y.pkl','rb'))
    model_rf = pickle.load(open('model_rf.pkl','rb'))
    
    json_file = open('model_nn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_nn = model_from_json(loaded_model_json)
    # load weights into new model
    model_nn.load_weights("model_nn.h5")
    #print("Loaded model from disk")
    opt = SGD(lr=0.1, momentum=0.9, decay=0.001)
    model_nn.compile(optimizer = opt, loss='sparse_categorical_crossentropy' ,  metrics=['accuracy'])
    
    json_file = open('model_ensemble.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_ensemble = model_from_json(loaded_model_json)
    # load weights into new model
    model_ensemble.load_weights("model_ensemble.h5")
    #print("Loaded model_ensemble from disk")
    model_ensemble.compile(optimizer = opt, loss='sparse_categorical_crossentropy' ,  metrics=['accuracy'])
    
    model_rf_input = model_rf.predict(TFIDF.transform([some_input]))
    model_nn_input = model_nn.predict(TFIDF.transform([some_input]))
    
    concated = []
    concated.append(np.concatenate((model_nn_input, model_nn_input), axis=1))
    concated = np.asarray(concated)
    concated = np.reshape(concated, (concated.shape[0], concated.shape[2]))
    
    #print(label_encoder_y.inverse_transform(np.argsort(model_ensemble.predict(concated), axis=1)[:,-1:][0])[0])
    k = 11
    model_rf_input = model_rf.predict_proba(TFIDF.transform([some_input]))
    model_nn_input = model_nn.predict(TFIDF.transform([some_input]))
    concated = []
    concated.append(np.concatenate((model_nn_input, model_rf_input), axis=1))
    concated = np.asarray(concated)
    concated = np.reshape(concated, (concated.shape[0], concated.shape[2]))
    
    worst_k = np.argsort(model_ensemble.predict(concated), axis=1)[:,-k:][0]
    #print('Companies to avoid:')
    worst = []
    for j in range(3):
        worst.append(label_encoder_y.inverse_transform([worst_k[k-1-j]])[0])
    #worst = np.asarray(worst)
    #worst = worst.reshape(3,1)
        
    best_k = np.argsort(model_ensemble.predict(concated), axis=1)[:,0:k][0]
    #print('\n','Companies for the project:')
    company = []
    mileage = []
    for j in range(k):
        comp = label_encoder_y.inverse_transform([best_k[j]])[0]
        if comp in mileage_df.name.values:
            company.append(comp)
            mileage.append(int(mileage_df.loc[mileage_df['name'] == comp, 'miles'].iloc[0]))
    df_best = pd.DataFrame({'Companies with least faiulres':company, 'Mileage': mileage})
    
    cleaned = re.sub('\W+', ' ', some_input)
    tokens = word_tokenize(cleaned)
    match = len([token for token in tokens if token in words])/len(tokens)
    
    return(df_best, worst, match)
    
def outputs_(some_input):
    return (pd.DataFrame([1,2], [2,3]), 'luck')

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
def home_page():
    return render_template('index_.html')

@app.route('/output')
def recommendation_output():
#       
       # Pull input
   some_input =str(request.args.get('user_input'))  
   df_best, worst, match = outputs(some_input)   

   # Case if empty
   if len(some_input)<35:
       return render_template("index_.html", 
                              my_input = some_input,
                              my_form_result="Empty")
   elif match < 0.2:
       return render_template("index_.html", 
                              my_input = some_input,
                              my_form_result="no_match")
           
   else:
       
       some_output="yeay!"
       some_number= 'Recommendation:'
       some_image="giphy.gif"
       return render_template("index_.html",tables=[df_best.to_html(classes='data')], titles=df_best.columns.values,
                          #my_input=label_encoder_y.inverse_transform(model_NLP.predict(TFIDF.transform([some_input])))[0],#some_input,
                          my_input = worst,
                          my_output=some_output,
                          my_number=some_number,
                          my_img_name=some_image,
                          my_form_result="NotEmpty")

# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(threaded=False)
    
    