from flask import Flask
from flask_restful import Api, Resource
from google import Create_Service
from flask_pymongo import PyMongo
from bson.json_util import dumps
import requests

app = Flask(__name__)
app.config['MONGO_URI']="" #MONGODB CONFIG #mongodb://your-domain/your-database-name

api=Api(app)
mongo=PyMongo(app) #mongo service object

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

#APPLICATION PROPERTIES
CLIENT_SECRET_FILE = ""# google drive api client secret file .json
API_NAME = "drive"
API_VERSION = "v3"
SCOPES = ["https://www.googleapis.com/auth/drive"]


FOLDER_ID = '' #google drive unknown classes folder id  #15705ceLk9E-i76GnBJUYzG-UINniZ6F4
DATA_FOLDER_ID='' #google drive dataset folder id #1AN8c_OeHETXlEKydevhmziPn5IyxYHmA


UNKNOWN_CLASS_COLLECTION=mongo.db.unknownclasses
KNOWN_CLASS_COLLECTION=mongo.db.knownclasses


DOMAIN="" #http://your-domain-name #http://127.0.0.1:5000

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

#CONTROLLER FUNCTIONS
def getService(CLIENT_SECRET_FILE,API_NAME,API_VERSION,SCOPES):
    service = Create_Service(CLIENT_SECRET_FILE,API_NAME,API_VERSION,SCOPES)
    return service

def getIdByName(image_name,folder_id):
    query=f"name='{image_name}' and parents in '{folder_id}' "
    response = service.files().list(q=query).execute()
    # print(response.get('files'))
    image_id=response.get('files')[0]['id']
    return image_id

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

#RESOURCE CLASSES

class CopyImage(Resource):
    def __init__(self, service): #Here service is taken from kwargs
        self.service = service #Initialize service

    def post(self,image_name):
        data_folder_id=DATA_FOLDER_ID
        folder_id=FOLDER_ID
        image_id=getIdByName(image_name,data_folder_id)

        file_metadata={
            'name':image_name,
            'parents':[folder_id],
        }

        response=service.files().copy(fileId=image_id,body=file_metadata).execute()
        new_file_id=response['id']

        #put in mongodb collection
        mongo_record_id=UNKNOWN_CLASS_COLLECTION.insert({"GoogleId":image_id,"UnknownClassId":new_file_id,"Name":image_name})

class GetImage(Resource):
    def __init__(self, service): #Here service is taken from kwargs
        self.service = service #Initialize service

    def get(self):
        folder_id = FOLDER_ID
        query = f"parents='{folder_id}'"
        response = service.files().list(q=query).execute()
        files=response.get('files')
        return files

class DeleteImage(Resource):
    def __init__(self, service): #Here service is taken from kwargs
        self.service = service #Initialize service
    
    def delete(self,image_id):
        response = service.files().delete(fileId=image_id).execute()
        return response, 204

class SetAnnotation(Resource):
    def __init__(self, service): #Here service is taken from kwargs
        self.service = service #Initialize service
    
    def post(self,image_id,class_name,is_new_class):
        mongo_record_id=UNKNOWN_CLASS_COLLECTION.update({"UnknownClassId":image_id},{'$set':{"class":class_name}})
        mongo_new_class_id="no new entry"
        if(int(is_new_class)):
            mongo_new_class_id=KNOWN_CLASS_COLLECTION.insert({"class_name":class_name,"class_count":1})
        response=requests.delete(DOMAIN+"/deleteimage/"+str(image_id))
        print("DELETING "+str(image_id))

        return {"updated_record":mongo_record_id}

class GetAnnotation(Resource):
    def __init__(self, service): #Here service is taken from kwargs
        self.service = service #Initialize service

    def get(self):
        response=UNKNOWN_CLASS_COLLECTION.find()
        annotations={}
        for document in response:
            annotations[document['GoogleId']]=document['class']
        return annotations

class GetClasses(Resource):
    def __init__(self, service): #Here service is taken from kwargs
        self.service = service #Initialize service
    
    def get(self):
        response=KNOWN_CLASS_COLLECTION.find()
        classes={}
        for document in response:
            classes[document['class_name']]=document['class_count']
        return classes
        
class Retrain(Resource):
    def __init__(self, service): #Here service is taken from kwargs
        self.service = service #Initialize service
    
    def post(self):
        #1) Get image ids and corrosponding annotations from mongo
        #2) Get encodings for those images in an array
        #3) Get encodings for those annotations in an array
        #4) Call ActiveLearningPreReq function --> create & define this in ActiveLearner.py
        #5) ActiveLearningPreReq will take encodings of images and annotations, it will then take the other necessary params such as Xtrain,Ytrain,Xtest,Ytest,model etc
        #and do the processing required which is adding the new encodings to Ytrain array
        #6) This function will then call and pass all the required params to ActiveLearner function where the training takes place
        #7) In ActiveLearner, the training will take place only once per function call and all the unknown images will then be copied into "Unknown Classes" folder 
        #in Drive, this is done by calling the CopyImage API and passing the image-id/image-name as parameter
        #8) Once the annotations for these unknown images is received, the process repeats
        print("in retrain")



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

#MAIN
if __name__=="__main__":

    #GET SERVICE
    service=getService(CLIENT_SECRET_FILE,API_NAME,API_VERSION,SCOPES)

    #ADD RESOURCE MAPPINGS
    api.add_resource(GetImage,"/getimages",resource_class_kwargs={'service': service} ) # PASS SERVICE PARAMETER AS KWARGS
    api.add_resource(DeleteImage,"/deleteimage/<string:image_id>",resource_class_kwargs={'service': service} ) # PASS SERVICE PARAMETER AS KWARGS
    api.add_resource(CopyImage,"/copyimage/<string:image_name>",resource_class_kwargs={'service': service} ) # PASS SERVICE PARAMETER AS KWARGS
    api.add_resource(SetAnnotation,"/setannotation/<string:image_id>/<string:class_name>/<string:is_new_class>",resource_class_kwargs={'service': service} ) # PASS SERVICE PARAMETER AS KWARGS
    api.add_resource(GetAnnotation,"/getannotation",resource_class_kwargs={'service': service} ) # PASS SERVICE PARAMETER AS KWARGS
    api.add_resource(GetClasses,"/getclasses",resource_class_kwargs={'service': service} ) # PASS SERVICE PARAMETER AS KWARGS

    app.run(debug=True)