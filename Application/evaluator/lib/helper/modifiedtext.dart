Future <void> main() async {
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