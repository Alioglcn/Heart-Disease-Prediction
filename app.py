import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

data = pd.read_csv('heart.csv.xls')

print(data.head())

selected_data = data[['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]

fig_scat = px.scatter(data, x="age", y="thalach", color="sex", title="Distribution Of Max Heart Rate By Age And Gender",size_max=30)
fig_scat.update_xaxes(title_text="Age")
fig_scat.update_yaxes(title_text="Maximum Heart Rate")
scat_graph_div = fig_scat.to_html(full_html=False, include_plotlyjs='cdn')




fig_hist = px.histogram(data, x='age', y="cp", color="cp", title="Distribution Of Chest Pain Severity By Age")
fig_hist.update_xaxes(title_text="Age")
fig_hist.update_yaxes(title_text="Chest of Pain")
hist_graph_div = fig_hist.to_html(full_html=False, include_plotlyjs='cdn')

fig_hist2 = px.density_heatmap(data, x="slope", y="oldpeak", text_auto=True, title="Heatmap of St Depression And Heart Rhythm Slope")
fig_hist.update_xaxes(title_text="The Slope Of The Peak Exercise St Segment")
fig_hist.update_yaxes(title_text="ST Depression Induced By Exercise")
hist_graph_div2 = fig_hist2.to_html(full_html=False, include_plotlyjs='cdn')

labels2 = {0: 'Not Carrier', 1: 'Thalassemia Carrier (Normal Level)', 2: 'Thalassemia Minor (Moderate Anemia)', 3: 'Thalassemia Major (Advanced Anemia)'}
data['thal'] = data['thal'].map(labels2)

fig_pie = px.pie(data, values='target', names='thal', title='The Effect Of Anemia On Heart Diseases', labels=labels2)
graph_div_pie = fig_pie.to_html(full_html=False, include_plotlyjs='cdn')

label = {0: 'No Pain', 1: 'Mild Pain', 2: 'Severe Pain', 3: 'Very Severe Pain'}
data['cp'] = data['cp'].map(label)

fig_pie2 = px.pie(data, values='ca', names='cp', color="target", title='The effect of anemia on heart diseases', labels=label)
box_graph_div = fig_pie2.to_html(full_html=False, include_plotlyjs='cdn')

fig_scat2 = px.scatter_3d(data, x="age", y="ca", z="target", color="target", title="Effect of chest pain on heart disease by age scale",size_max=30)
fig_hist.update_xaxes(title_text="a")
fig_hist.update_yaxes(title_text="b")

scat_graph_div2 = fig_scat2.to_html(full_html=False, include_plotlyjs='cdn')

# Create an app object using the Flask class.
app = Flask(__name__)

# Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

# Define the route to be home.
# The decorator below links the relative route of the URL to the function it is decorating.
# Here, home function is with '/', our root directory.
# Running the app sends us to index.html.
# Note that render_template means it looks for the file in the templates folder.

# use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/portfolio-details-4.html')
def portfolio4():
    return render_template('portfolio-details-4.html')

@app.route('/portfolio-details.html')
def portfolio():
    return render_template('portfolio-details.html')

@app.route('/portfolio-details2.html')
def portfolio2():
    return render_template('portfolio-details2.html')

@app.route('/portfolio-details-3.html')
def portfolio3():
    return render_template('portfolio-details-3.html')

@app.route('/blog.html')
def blog():
    return render_template('blog.html')

@app.route('/details.html')
def details():
    return render_template('details.html')


# You can use the methods argument of the route() decorator to handle different HTTP methods.
# GET: A GET message is sent, and the server returns data
# POST: Used to send HTML form data to the server.
# Add Post method to the decorator to allow for form submission.
# Redirect to /predict page with the output
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    image_url = 'gif{}.gif'.format(int(prediction))

    if prediction == 0:
       prediction_text = "Yes"
    elif prediction == 1:
        prediction_text = "No"


    return render_template('details.html', prediction_text=prediction_text, prediction=int(prediction), image_url=image_url, features=dict(zip(['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'], int_features)), hist_graph_div=hist_graph_div, hist_graph_div2=hist_graph_div2, graph_div_pie=graph_div_pie, scat_graph_div=scat_graph_div, box_graph_div=box_graph_div, scat_graph_div2=scat_graph_div2)


# When the Python interpreter reads a source file, it first defines a few special variables.
# For now, we care about the __name__ variable.
# If we execute our code in the main program, like in our case here, it assigns
#
if __name__ == "__main__":
    app.run()       