from flask import Flask, jsonify
import flask
#from flask_restful import Api, Resource, reqparse
import pandas as pd
import app.myModel as myModel
import json

app = Flask(__name__)
# api = Api(app)  
path = 'app/dataset/'
dbClientModel = myModel.CSV_DataBase(path + 'application_test.csv')
dbClientPrets = myModel.CSV_DataBase(path + 'bureau.csv')
mymodel = myModel.Model(path = path, database = dbClientModel)

myModel.log.info("ready to start")


@app.route('/api/v1/client_info/<client_id>', methods=['GET'])
def get_client_infos(client_id:int):
    client_id = int(client_id)
    myModel.log.info(f"get infos for {client_id}")
    infos = myModel.DataBase.DataFrame2Json(dbClientModel.get_id_client(client_id))
    myModel.log.info(infos)
    return flask.jsonify(infos)
    #return json.dumps(infos)

@app.route('/api/v1/prediction/<client_id>', methods=['GET'])
def get_client_prediction(client_id:int):
    myModel.log.info(f"get prediction for {client_id}")
    return flask.jsonify(mymodel.predict_id(int(client_id)).tolist()[0])

@app.route('/api/v1/group_info/<client_id>', methods=['GET'])
def get_client_group(client_id:int):
    myModel.log.info(f"get infor for {client_id} group")
    return flask.jsonify(mymodel.predict_id(int(client_id)).tolist()[0])

@app.route('/api/v1/client_prets/<client_id>', methods=['GET'])
def get_client_prets(client_id:int) :
    client_id = int(client_id)
    myModel.log.info(f"get prets for {client_id}")
    infos = myModel.DataBase.DataFrame2Json(dbClientPrets.get_id_client(client_id))
    myModel.log.debug(infos)
    return flask.jsonify(infos)



if __name__ == '__main__':
    app.run(debug=True)
