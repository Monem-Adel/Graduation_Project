# import cv2
# from PIL import Image

# method to convert image to binary
# the parameter image is the path of the image
# def convertToBinary(image):
#     # read the image from the path in gray scale color system
#     # cv2.IMREAD_GRAYSCALE ≡ 2
#     img = cv2.imread(image,cv2.IMREAD_GRAYSCALE) # return a numpy object

#     # convert to binary depending on the threshold value
#     _ , binaryImg = cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)
#     return binaryImg

# im = cv2.imread(r'H:\Scanner\00.jpg')
# im = convertToBinary(r'H:\Scanner\00.jpg')
# to convert from numPy to image
# im2 = Image.fromarray(im) 
# im2.show()
# cv2.imshow(window_name, image)
# cv2.imshow('binary image',im)
# cv2.waitKey(0) 

# def get_head_tail(img):
#     img = convertToBinary(img)
#     pass



# lst = ['s1','s2','s3','s4','s5','s6']
# st={'1','2','3','4'}
# table = [['#', *list(st)],['ddd','11'],['fff','22'],['ggg','2165']]
# #sliced_list = original_list[start_row:end_row:step_row][start_col:end_col:step_col]
# for row , item in zip(table[1:],range(len(lst))):
#    row[0]=lst[item]
# print(len(table))
# print(len(table[1:]))
# N=len(table[1:])
# print(len(lst)-len(table[1:]))
# print(range(abs(len(lst)-len(table[1:]))))
# for item in range(len(lst)-len(table[1:])):
#     table.append([lst[item+N]])
#     print(item)
# print(table)
# # table[4].append('ggf')
# # print(table[4][1])
# print(table[4][0])
# table[4].insert(1,'mmmmmmm')
# table[4][1]='0000'
# # table[3][1].insert(0,'fff')
# print (table)

# lst=[1,2,3,4,5,6,7,8,9]
# for i in lst:
#     i+=9
# print(lst)

# print(*lst)
# print(*list(st))

# Initializing list
# lst = [1, 6, 3,3, 5,4]

# s = next((i for i in lst if i%2==0),'nn')
# print(s)

# my_dict = {'a': 1, 'b': 2, 'c': 3}

# print(my_dict.get('n'))

# s = {}
# s.add(1)
# s.add(2)
# s.add(1)
# print(s)
# dic = {'one':1 , 'two':2,'three':3}
# lst = [{'one':1 , 'two':2,'three':3},{'one':4 , 'two':5,'three':6},{'one':7 , 'two':8,'three':9}]
# for i  in lst:
#     i1 = i['one']
#     i2 = i['two']
#     i3 = i['three']
#     print(i1)
#     print(i2)
#     print(i3)

# lst = []
# print(len(lst))

import os 
lst = ['1']
lst.extend(os.listdir(r'H:\Graduation Project\Graduation_Project'))
print(lst)