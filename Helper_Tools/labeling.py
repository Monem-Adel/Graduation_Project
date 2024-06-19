import shutil
import PIL
from pigeon import annotate
from IPython.display import display, Image
import os

#Take the path of the file dataset to labeling
inputPath = "F:/عبدالمنعم/Academic/Level 4/Graduation Project/Dataset/Basis"
files = os.listdir(inputPath)

#List of paths for each file in files
files_paths = [os.path.join(inputPath, file) for file in files]

#List of buttons (labels)
btnLst = ['Start State', 'State', 'Final State', 'Transition', 'Alphabet', 'Ignore']

#Annotation function to label each file
labels = annotate(
    files_paths,
    options=btnLst,
    display_fn=lambda filename: display(Image(filename))
)
#print(labels)

#define a function to splitng the labels
def makeLabels(L, p):
    output = p

    #Create the output folder if it doesn't exist
    if not os.path.exists(output):
        os.makedirs(output)

    #Create a directory for each label
    folders = ['Start_State', 'State', 'Final_State', 'Transition', 'Alphabet', 'Ignore']
    for folder in folders:
        os.makedirs(os.path.join(output, folder))

    #Labeling
    for item in L:
        pth_item, label = item
        #shutil.copy(source_file_path, destination_file_path)
        if label == 'Start State':
            shutil.copy(pth_item, os.path.join(output, 'Start_State'))
        elif label == 'State':
            shutil.copy(pth_item, os.path.join(output, 'State'))
        elif label == 'Final State':
            shutil.copy(pth_item, os.path.join(output, 'Final_State'))
        elif label == 'Transition':
            shutil.copy(pth_item, os.path.join(output, 'Transition'))
        elif label == 'alphabet':
            shutil.copy(pth_item, os.path.join(output, 'alphabet'))
        else:
            shutil.copy(pth_item, os.path.join(output, 'Ignore'))


#to make the labeled img's files call this method with specified path
#makeLabels(labels_got_from_annotation , the output path)

