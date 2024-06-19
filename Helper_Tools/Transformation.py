from PIL import Image, ImageEnhance, ImageFilter
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T

#add the path of input folder & output folder
input ="F:\عبدالمنعم\Academic\Level 4\Graduation Project\Dataset\Basis" #حط هنا الباث بتاع الداتا ست
transformated_images ="F:\عبدالمنعم\Academic\Level 4\Graduation Project\Dataset\\transformated_images" #output directory

#Create the output folder if it doesn't exist
if not os.path.exists(transformated_images):
    os.makedirs(transformated_images)

#get all files in the input folder
files = os.listdir(input)

#processing each image in the folder
i=0
for file in files :
    #check if the file is an img or not
    if not file.lower().endswith(('.jpeg', '.jpg', '.png')):
        continue  # skip if not an image file
    
    img = Image.open(os.path.join(input,file))

    #add blur filter
    newIm=img.filter(ImageFilter.BLUR)
    newIm.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1
    
    #GaussianBlur
    newIm2=T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
    newIm2.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1

    #grayscale
    newIm3=T.Grayscale(1)(img)
    newIm3.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1

    #RandomEqualize
    newIm4 = T.RandomEqualize()(img)
    newIm4.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1

    #sharpen
    newIm5=T.RandomAdjustSharpness(sharpness_factor=3)(img)
    newIm5.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1

    #contrast
    newIm6=T.RandomAutocontrast()(img)
    newIm6.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1

    #Solarize
    newIm7=T.RandomSolarize(threshold=192.0)(img)
    newIm7.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1

    #colorjitter
    newIm8=T.ColorJitter(brightness=5, contrast=3, saturation=6, hue=0.1)(img)
    newIm8.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1
    
    #resize
    newIm9 = T.Resize(size=300)(img)
    newIm9.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1

    #elastic
    newIm10 = T.ElasticTransform(alpha=100.0)(img)
    newIm10.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1

    #combination
    newIm11=img.filter(ImageFilter.BLUR)
    newIm11=T.Grayscale(1)(newIm11)
    newIm11=T.RandomAdjustSharpness(sharpness_factor=3)(newIm11)
    newIm11=T.RandomAutocontrast()(newIm11)
    newIm11=T.ColorJitter(brightness=5, contrast=3, saturation=6, hue=0.1)(newIm11)
    newIm11=T.RandomSolarize(threshold=192.0)(newIm11)
    newIm11.save(os.path.join(transformated_images, f'transformed_img_{i}.jpg'))
    i+=1

    # plt.imshow(newIm4)
    # plt.show()
