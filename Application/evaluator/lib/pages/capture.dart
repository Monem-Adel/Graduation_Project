import 'dart:convert';
import 'dart:io';
import 'package:evaluator/component/cust_buttom.dart';
import 'package:evaluator/component/cust_textfield.dart';
import 'package:evaluator/pages/result.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class Capture extends StatefulWidget {
  Capture({super.key});

  @override
  State<Capture> createState() => _CaptureState();
}
File? imageFile;
String? testCase;
String result="";
class _CaptureState extends State<Capture> {
  bool isImageSelected = false;

  //File? _selectedImage;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      bottomNavigationBar: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          GestureDetector(
            onTap: () async {
              await FirebaseAuth.instance.signOut();
              Navigator.pop(context);
            },
            child: Icon(Icons.exit_to_app,color: Colors.white,)
          ),
        ],
      ),
      backgroundColor: Color(0xff22B14C),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 120,),
        child: ListView(
            children: [
              Spacer(flex: 1,),
              Image(image: AssetImage('assets/images/scholar.jpg')),
              Text(
                'Your Evaluator ',
                style: TextStyle(
                    color: Colors.white54,
                    fontSize:22,
                    fontWeight: FontWeight.bold),
              ),
              SizedBox(
                height: 35,
              ),
              TextField(
                decoration: InputDecoration(
                  hintText: '   Enter a test case',
                ),
                onChanged:(value) {
                  //print("moosss=$value");
                  testCase=value;// this variable - testCase - used to catch the text and use it when ever you want
                  //print("moosss=$testCase");
                },onSubmitted: (value) {
                //print("moosss=$value");
                setState(() {
                  testCase=value;
                });// this variable - testCase - used to catch the text and use it when ever you want
                //print("moosss=$testCase");
              },
              ),
              SizedBox(height: 10,),
              GestureDetector(
                onTap: (){
                  _pickImagefromCamera();
                },
                child: Custom_Button(
                  width: 120,
                  text: 'Capture image',
                ),
              ),
              SizedBox(height: 10,),
              GestureDetector(
                onTap: (){
                  _pickImagefromGallrey();
                },
                child: Custom_Button(
                  width: 120,
                  text: 'Select image',
                ),
              ),
              SizedBox(height: 10,),
              GestureDetector(
                onTap: (){
                  Upload_Image_and_Text();
                  Navigator.push(context, MaterialPageRoute(builder: (context){
                    return Result();
                  }));
                },
                child: Custom_Button(
                  width: 120,
                  text: 'Upload',
                ),
              ),
              SizedBox(height: 10,),
              Column(
                children: [
                  Center(
                    child: imageFile==null ? Text('plz write and pick then'):  // try this widget
                    Image.file( imageFile!, ),
                  ),
                  if(testCase!=null)
                    Text(testCase!)
                ],
              ),
            ]
          ),
        ),
    );
  }
  _pickImagefromCamera() async {
    try {
      final pickedImage =
      await ImagePicker().pickImage(source: ImageSource.camera,maxHeight: 20,maxWidth: 20);
      if (pickedImage != null) {
        setState(() {
          imageFile = File(pickedImage.path);
          isImageSelected = true;
        });
      } else {
        print('User didnt pick any image.');
      }
    } catch (e) {
      print(e.toString());
    }
  }
  _pickImagefromGallrey() async {
    try {
      final pickedImage =
      await ImagePicker().pickImage(source: ImageSource.gallery);
      if (pickedImage != null) {
        setState(() {
          imageFile = File(pickedImage.path);
          isImageSelected = true;
        });
      } else {
        print('User didnt pick any image.');
      }
    } catch (e) {
      print(e.toString());
    }
  }
  Upload_Image_and_Text() async {
    String? text = testCase;
    final request = http.MultipartRequest(
        'POST',
        Uri.parse(
            "http://192.168.1.4:5000/api")); /*here you should write your  public ip address */
    request.fields["str"] = text!;
    request.files.add(http.MultipartFile('image',
        imageFile!.readAsBytes().asStream(), imageFile!.lengthSync(),
        filename: imageFile!.path.split("/").last));
    final response = await request.send();
    http.Response res = await (http.Response.fromStream(response));
    var resJson = jsonDecode(res.body);
    setState((){
      result=res as String;
      print("res=${resJson['result']}");
    });
  }

  modifiedtheText(){
    String? testCase = "This is a test case"; // Example input
    String? originalString = testCase;

    // Check if originalString is null to avoid null pointer exceptions
    if (originalString != null) {
      String? modifiedString = originalString.replaceAll(' ', ':');
      String str = "";
      List<String> tstcase = [];

      for (int i = 0; i < modifiedString.length; i++) {
        if (modifiedString[i] != ':') {
          str += modifiedString[i];
        } else {
          // Add the accumulated string to the list
          tstcase.add(str);
          // Reset the str to accumulate next substring
          str = "";
        }
      }

      // Add the last accumulated substring 4 times (if any)
      if (str.isNotEmpty) {
        tstcase.add(str);
      }

      print(tstcase);
    }
  }
}
