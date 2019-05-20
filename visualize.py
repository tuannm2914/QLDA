"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
import cv2

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_face_image(image):
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )

  # None is no face found in image
  if not len(faces) > 0:
    return None, None
  max_are_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face
  # face to image
  face_coor =  max_are_face
  image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]

  return  image, face_coor

def format_image(image):
  gray = rgb2gray(image)
  gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

  img = gray[:, :, np.newaxis]

  img = np.concatenate((img, img, img), axis=2)
  img = Image.fromarray(img)
  inputs = transform_test(img)
  ncrops, c, h, w = np.shape(inputs)

  inputs = inputs.view(-1, c, h, w)
  # inputs = inputs.cuda()
  inputs = Variable(inputs, volatile=True)
  return inputs,ncrops


net = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'),map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint['net'])
#net.cuda()
net.eval()

def demo(showBox = True):


  feelings_faces = []
  for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./images/emojis/' + emotion + '.png', -1))
  video_captor = cv2.VideoCapture(0)

  emoji_face = []
  predicted = None

  while True:
    ret, frame = video_captor.read()
    if ret == True:
      frame = cv2.flip(frame, 1)

      detected_face, face_coor = get_face_image(frame)
      if showBox:
        if face_coor is not None:
          [x, y, w, h] = face_coor
          cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

      if cv2.waitKey(1) & 0xFF == ord(' '):

        if detected_face is not None:
          cv2.imwrite('a.jpg', detected_face)
          inputs,ncrops = format_image(detected_face)
          outputs = net(inputs)

          outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

          score = F.softmax(outputs_avg)
          _, predicted = torch.max(outputs_avg.data, 0)
          # print(result)
      if predicted is not None:
        for index, emotion in enumerate(EMOTIONS):

          emoji_face = feelings_faces[int(predicted.cpu().numpy())]

        for c in range(0, 3):
          frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130,
                                                                                            c] * (
                                                1.0 - emoji_face[:, :, 3] / 255.0)
      cv2.imshow('face', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break


  print(int(predicted.cpu().numpy()))

"""
plt.rcParams['figure.figsize'] = (13.5,5.5)
axes=plt.subplot(1, 3, 1)
plt.imshow(raw_img)
plt.xlabel('Input Image', fontsize=16)
axes.set_xticks([])
axes.set_yticks([])
plt.tight_layout()


plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

plt.subplot(1, 3, 2)
ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
width = 0.4       # the width of the bars: can also be len(x) sequence
color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
for i in range(len(class_names)):
    plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
plt.title("Classification results ",fontsize=20)
plt.xlabel(" Expression Category ",fontsize=16)
plt.ylabel(" Classification Score ",fontsize=16)
plt.xticks(ind, class_names, rotation=45, fontsize=14)

axes=plt.subplot(1, 3, 3)
emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
plt.imshow(emojis_img)
plt.xlabel('Emoji Expression', fontsize=16)
axes.set_xticks([])
axes.set_yticks([])
plt.tight_layout()
# show emojis

#plt.show()
plt.savefig(os.path.join('images/results/l.png'))
plt.close()

print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
"""

if __name__ == "__main__":
  demo()