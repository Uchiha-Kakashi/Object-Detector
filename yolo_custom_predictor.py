import numpy as np
import cv2

class Yolodetector(object):

    def __init__(self, img_path):
        self.img_path = img_path

    def detector(self):
        confidenceThreshold = 0.5
        NMSThreshold = 0.3

        modelConfiguration = 'cfg/yolov3_custom.cfg'
        modelWeights = 'weights/yolov3_custom.weights'

        labelsPath = 'coco_custom.names'
        labels = open(labelsPath).read().strip().split('\n')

        np.random.seed(10)
        COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

        image = cv2.imread(self.img_path)
        (H, W) = image.shape[:2]

        #Determine output layer names
        layerName = net.getLayerNames()
        layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
        net.setInput(blob)
        layersOutputs = net.forward(layerName)

        boxes = []
        confidences = []
        classIDs = []

        for output in layersOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidenceThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY,  width, height) = box.astype('int')
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        #Apply Non Maxima Suppression
        detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

        outClasses = []
        if (len(detectionNMS) > 0):
            for i in detectionNMS.flatten():
                outClasses.append(labels[classIDs[i]])
                # print(labels[classIDs[i]])

        if(len(detectionNMS) > 0):
            for i in detectionNMS.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # Saving the predicted image in the directory
        img_name = self.img_path.split('/')[-1]
        predictPath = 'static/Images/Predicted/{}'.format(img_name)
        cv2.imwrite(predictPath, image)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        return outClasses, predictPath


if __name__ == '__main__':
    a = Yolodetector('static/Images/Uploaded/wheelchair.jpg')
    a.detector()
