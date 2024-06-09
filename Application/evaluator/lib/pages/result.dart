import 'dart:convert';
import 'package:evaluator/pages/capture.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:modal_progress_hud_nsn/modal_progress_hud_nsn.dart';
import '../component/cust_buttom.dart';
import '../component/cust_textfield.dart';
import '../helper/show_snack_bar.dart';

class Result extends StatefulWidget {
  Result({super.key});

  get passwordInVisible => true;

  @override
  State<Result> createState() => _ResultState();
}

class _ResultState extends State<Result> {
  GlobalKey<FormState> formkey = GlobalKey();
  bool isload = false;
  String? email1;
  String? password1;
  String json="";
  @override
  Widget build(BuildContext context) {
    return ModalProgressHUD(
        inAsyncCall: isload,
        child: Scaffold(
          backgroundColor: Color(0xff8399A8),
          body: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Form(
              key: formkey,
              child: ListView(children: [
                SizedBox(
                  height: 65,
                ),
                Image(
                  image: AssetImage('assets/images/scholar.jpg'),
                  width: 100,
                  height: 125,
                ),
                SizedBox(
                  height: 15,
                ),
                Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                  Text(
                    'Your Evaluator ',
                    style: TextStyle(
                        color: Colors.white54,
                        fontSize: 22,
                        fontWeight: FontWeight.bold),
                  ),
                ]),
                SizedBox(height: 20,),
                Text(
                  textAlign: TextAlign.center,
                  testCase!,
                  style: TextStyle(
                      color: Colors.black87,
                      fontSize: 22,
                      fontWeight: FontWeight.bold),
                ),
                Image.file(imageFile!,
                height: 230,
                width: 230,)
              ]),
            ),
          ),
        ));
  }

}
