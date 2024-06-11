import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:modal_progress_hud_nsn/modal_progress_hud_nsn.dart';
import '../component/cust_buttom.dart';
import '../component/cust_textfield.dart';
import '../helper/show_snack_bar.dart';

class Login extends StatefulWidget {
  Login({super.key});

  get passwordInVisible => true;

  @override
  State<Login> createState() => _LoginState();
}

class _LoginState extends State<Login> {
  GlobalKey<FormState> formkey = GlobalKey();
  bool isload = false;
  String? email1;
  String? password1;
  @override
  Widget build(BuildContext context) {
    return ModalProgressHUD(
      inAsyncCall: isload,
      child: Scaffold(
          backgroundColor: Color(0xff22B14C),
          body: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Form(
              key: formkey,
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
                            color: Colors.black87,
                            fontSize: 22,
                            fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Row(
                    children: [
                      Text(
                        'LOGIN',
                        style: TextStyle(
                            color: Colors.black,
                            fontSize: 28,
                          ),
                      ),
                    ],
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Custom_TextFormField(
                    onChange: (data) {
                      email1 = data;
                    },
                    passwordInVisible: false,
                    hintText: 'Email:',
                    ic: Icons.email_outlined,
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Custom_TextFormField(
                    onChange: (data) {
                      password1 = data;
                    },
                    passwordInVisible: true,
                    hintText: 'Password:',
                    ic: Icons.lock_open,
                  ),
                  SizedBox(
                    height: 20,
                  ),
                  GestureDetector(
                    onTap: () {},
                    child: Custom_Button(
                      onTap: () async {
                        if (formkey.currentState!.validate()) {
                          isload = true;
                          setState(() {});
                          try {
                            await LoginUser();
                            showSnackBar(context, 'Success!!');
                            Navigator.pushNamed(context, 'Capture');
                          } on FirebaseAuthException catch (e) {
                            User? user = FirebaseAuth.instance.currentUser;

                            if (user!= null && !user.emailVerified) {
                              await user.sendEmailVerification();
                              showSnackBar(context, 'invalid email or password');
                            }
                          } catch (e) {
                            showSnackBar(context, 'there is an error');
                          }
                          isload = false;
                          setState(() {});
                        } else {}
                      },
                      text: 'sign in',
                    ),
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'don\'t have an account? ',
                        style: TextStyle(color: Colors.black),
                      ),
                      GestureDetector(
                        onTap: () {
                          Navigator.pushNamed(context, 'Registration');
                        },
                        child: Container(
                          child: Text(
                            'SignUp',
                            style: TextStyle(color: Colors.white54),
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

  Future<void> LoginUser() async {
    UserCredential user = await FirebaseAuth.instance
        .signInWithEmailAndPassword(email: email1!, password: password1!);
  }
}
