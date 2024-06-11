import 'dart:ui';

import 'package:flutter/material.dart';

class Custom_Button extends StatelessWidget {
  // ignore: use_key_in_widget_constructors
  Custom_Button({this.text, this.width, this.onTap});
  VoidCallback? onTap;
  String? text;
  double? width;
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 40,
        width: width,
        decoration: BoxDecoration(
          color: Colors.black,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Center(
            child: Text(
          text!,
              style: TextStyle(color:Color(0xff22B14C)),
        ),
        ),
      ),
    );
  }
}
