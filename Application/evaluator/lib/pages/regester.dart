import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:modal_progress_hud_nsn/modal_progress_hud_nsn.dart';
import '../component/cust_buttom.dart';
import '../component/cust_textfield.dart';
import '../helper/show_snack_bar.dart';

class Register extends StatefulWidget {
  Register({super.key});

  get passwordInVisible => false;

  @override
  State<Register> createState() => _RegisterState();
}

class _RegisterState extends State<Register> {
  String? email;

  String? password;

  String? confirmpassword;

  GlobalKey<FormState> formkey =GlobalKey();

  final TextEditingController _pass = TextEditingController();


  bool isLoad =false;

  @override
  Widget build(BuildContext context) {
    return ModalProgressHUD(
      inAsyncCall: isLoad,
      child: Scaffold(
          backgroundColor: Color(0xff8399A8),
          body: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Form(
              key:formkey ,
              child: ListView(
                children: [
                  SizedBox(
                    height: 65,
                  ),
                  Image(image: AssetImage('assets/images/scholar.jpg'),
                    width: 100,
                    height: 125,),
                  SizedBox(
                    height: 15,
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Your Evaluator ',
                        style: TextStyle(
                            color: Colors.white54,
                            fontSize: 22,
                            fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      Text(
                        'Sign Up',
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 28,
                            fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Custom_TextFormField(
                    passwordInVisible: false,
                    onChange: (data) {
                      email = data;
                    },
                    hintText: 'Email:',
                    ic: Icons.email_outlined,
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Custom_TextFormField(
                    passwordInVisible: true,
                    onChange: (data) {
                      setState(() {
                        password = data;
                      });
                    },
                    hintText: 'Password:',
                    ic: Icons.lock_open,
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Custom_TextFormField(
                    passwordInVisible: true,
                    onChange: (data) {
                      setState(() {
                        confirmpassword =data ;

                      });
                    },
                    hintText: 'Confirm Password:',
                    ic: Icons.lock_open,
                  ),// confirm pasword
                  SizedBox(
                    height: 20,
                  ),
                  GestureDetector(
                    onTap: () {
                    },
                    child: Custom_Button(
                      onTap: () async {
                        if (formkey.currentState!.validate()) {
                          isLoad=true;
                          try{
                            if(password==confirmpassword){
                              await RegesterUser();
                              showSnackBar(context, 'Success!!');
                              Navigator.pop(context);
                            }else{
                              showSnackBar(context, 'password != confirmpass');
                            }
                          }on FirebaseAuthException catch (e) {
                            if (e.code == 'weak-password') {
                              showSnackBar(context,'weak password');
                            } else if (e.code == 'email-already-in-use') {
                              showSnackBar(context, 'email already exists');
                            }
                          }catch(e){
                            showSnackBar(context, 'there is an error');
                          }
                          isLoad=false;
                          setState(() {});
                        }
                        else{

                        }
                      },
                      text: 'sign up',
                    ),
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Already have an account? ',
                        style: TextStyle(color: Colors.white),
                      ),
                      GestureDetector(
                        onTap: () {
                          Navigator.pushNamed(context, 'LogIn');
                        },
                        child: Container(
                          child: Text(
                            'Sign In',
                            style: TextStyle(color: Colors.white38),
                          ),
                        ),
                      )
                    ],
                  ),
                ],
              ),
            ),
          )),
    );
  }
  Future<void> RegesterUser() async {
    UserCredential user = await FirebaseAuth.instance
        .createUserWithEmailAndPassword(
        email: email!, password: password!);
  }
}
