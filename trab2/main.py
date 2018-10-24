from CNN import CNN
import cv2


names = {0: 'airplane',
         1: 'automobile',
         2: 'bird',
         3: 'cat',
         4: 'deer',
         5: 'dog',
         6: 'frog',
         7: 'horse',
         8: 'ship',
         9: 'truck'}

if __name__ == '__main__':
    num_classes = 10
    path = 'test_images/'
    img_size = 32
    model = CNN(num_classes)
    # model.load('model.h5')
    images = []
    for key, value in names.items():
        img = cv2.imread(path + value + '.jpeg', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
    images = [image/255 for image in images]
