import 'dart:io';
import 'package:evaluator/component/cust_buttom.dart';
import 'package:evaluator/component/cust_textfield.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class Capture extends StatefulWidget {
  Capture({super.key});

  @override
  State<Capture> createState() => _CaptureState();
}

class _CaptureState extends State<Capture> {
  bool isImageSelected = false;
  File? imageFile;
  String? testCase;
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
      backgroundColor: Color(0xff8399A8),
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
                  print("moosss=$value");
                  testCase=value;// this variable - testCase - used to catch the text and use it when ever you want
                  print("moosss=$testCase");
                },onSubmitted: (value) {
                print("moosss=$value");
                setState(() {
                  testCase=value;
                });// this variable - testCase - used to catch the text and use it when ever you want
                print("moosss=$testCase");
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
              Spacer(flex: 2,),
              SizedBox(height: 100,),
              Column(
                children: [
                  Center(
                    child: imageFile==null ? Text('plz pick'):  // try this widget
                    Image.file( imageFile!, ),
                  ),
                  if(testCase!=null)
                    Text(testCase!)
                ],
              ),
              SizedBox(height: 100,),
              GestureDetector(
                onTap: (){
                  Navigator.pushNamed(context, 'Result');
                },
                child: Custom_Button(
                  width: 120,
                  text: 'Show result',
                ),
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
    String text = _controller.text;
    final request =
        http.MultipartRequest('POST', Uri.parse("http://192.168.1.9:5000/api")); /*here you should write your  public ip address */
    //final headers = {"Content-tyoe": "multipart/form-data"};
    request.fields["str"] = text;
    request.files.add(http.MultipartFile('image',
        selectedImage!.readAsBytes().asStream(), selectedImage!.lengthSync(),
        filename: selectedImage!.path.split("/").last));
    //request.headers.addAll(headers);
    final response = await request.send();
    http.Response res = await (http.Response.fromStream(response));
    final resJson = jsonDecode(res.body);
    print(resJson['result']);
    setState(() {});
  }

}
