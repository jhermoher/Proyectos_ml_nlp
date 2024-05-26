from flask import Flask
from flask_restx import Api, Resource, fields, reqparse
import joblib
from flask_cors import CORS
import os

from team08_model_prediction2 import predictions

app = Flask(__name__)
CORS(app) 

model_genre_clf = joblib.load(os.path.join(os.path.dirname(__file__), 'model_genre_clf_1.pkl'))
model_genre_clf.classes_ = joblib.load(os.path.join(os.path.dirname(__file__), 'model_genre_clf_1_classes_.pkl'))
le = joblib.load(os.path.join(os.path.dirname(__file__), 'label_encoder.pkl'))

api = Api(
    app, 
    version='0.0c', 
    title='Movie Genres Prediction API',
    description='API for prediction of the genres of movies based on their plot. Developed by Team 8.')

ns = api.namespace('Prediction', description='Movie Genre Predictor')

resource_fields = api.model('Resource', {'result': fields.String})

parser = reqparse.RequestParser()
parser.add_argument(
    'Movie Plot', 
    type=str, 
    required=True, 
    help='Plot text to be analyzed', 
    location='args')

@ns.route('/')  
class Predict(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()
        plot_text = args['Movie Plot']
        result = predictions(plot_text)
        return {'result': result}, 200
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
