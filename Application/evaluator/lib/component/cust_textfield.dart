import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart'
    show BorderSide, BuildContext, Colors, IconButton, Icons, InputDecoration, OutlineInputBorder, StatelessWidget, TextField, TextFormField, TextStyle, Theme, Widget;

// ignore: camel_case_types
class Custom_TextFormField extends StatefulWidget {
  // ignore: use_key_in_widget_constructors
  Custom_TextFormField({this.hintText,this.ic,this.sc, required this.passwordInVisible,this.onChange});
  String? hintText;
  IconData? ic;
  IconButton? sc;
  Function(String)? onChange;
  bool passwordInVisible;
  final TextEditingController _pass = TextEditingController();
  @override
  State<Custom_TextFormField> createState() => _Custom_TextFormFieldState();
}

class _Custom_TextFormFieldState extends State<Custom_TextFormField> {
  @override
  Widget build(BuildContext context) {
    return TextFormField(
      obscureText: widget.passwordInVisible,
      validator: (data) {
        if(data!.isEmpty){
          return 'Field is required';
        }

        },

      onChanged: widget.onChange ,
      decoration: InputDecoration(
            prefixIcon: Icon(color: Colors.white30, widget.ic),
            hintText: widget.hintText,
            hintStyle: const TextStyle(
            color: Colors.white30,
          ),
          enabledBorder: const OutlineInputBorder(
              borderSide: BorderSide(color: Colors.white)),
          border: const OutlineInputBorder(borderSide: BorderSide()),),
    );
  }
}
