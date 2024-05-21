from flask import Flask, jsonify, request
import requests
import werkzeug
import werkzeug.utils

app = Flask(__name__)






@app.route('/api', methods=['POST'])
def evaluate():
    if request.method == "POST":
        # the following part is to recieve the image from flutter app request and save it in Automatonimage folder
        image_file = request.files['image']
        file_name = werkzeug.utils.secure_filename(image_file.filename)
        image_file.save("./AutomatonImage/"+file_name)
        # now, recieving a string
        s1 = str (request.args['str'])
        #res = evaluate(sl) // later, to test function call on the input and get the results 
        json_file ={}
        json_file['result'] = 'the image has been recieved succefully and the string sent is  is '+str(s1) 
        return jsonify(json_file)



if __name__ == '__main__':
    app.run()




