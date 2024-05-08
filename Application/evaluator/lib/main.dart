import 'package:evaluator/pages/capture.dart';
import 'package:evaluator/pages/login.dart';
import 'package:evaluator/pages/regester.dart';
import 'package:evaluator/pages/result.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
Future <void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(Corrector());
}

class Corrector extends StatefulWidget {
  const Corrector({super.key});

  @override
  State<Corrector> createState() => _CorrectorState();
}

class _CorrectorState extends State<Corrector> {
  @override
  void initState() {
    FirebaseAuth.instance
        .authStateChanges()
        .listen((User? user) {
      if (user == null) {
        print('==================================================================User is currently signed out!');
      } else {
        print('==================================================================User is signed in!');
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      routes: {
        'LogIn' : (context)=> Login(),
        'Registration': (context)=> Register(),
        'Capture' : (context)=> Capture(),
        'Result' : (context) => Result(),
      },
      initialRoute: 'LogIn',
      debugShowCheckedModeBanner: false,
    );
  }
}
