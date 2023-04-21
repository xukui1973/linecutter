import cv2
import copy
from collections import defaultdict

#def list_duplicates(seq):
    #tally = defaultdict(list)
    #for i,item in enumerate(seq):
       # tally[item].append(i)
  #  return ((key,locs) for key,locs in tally.items() 
                           # if len(locs)>1)

def in_nested_list(my_list, item):
    if item in my_list:
        return True
    else:
        return any(in_nested_list(sublist, item) for sublist in my_list if isinstance(sublist, list))
#trackers = cv2.legacy.MultiTracker_create()
needQuit=0
video = cv2.VideoCapture(0)
video.set(3, 640)
video.set(4, 420)
faceCascade = cv2.CascadeClassifier("/anaconda3/envs/py38/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
#faceCascade = cv2.CascadeClassifier("/anaconda3/envs/py38/share/opencv4/haarcascades/haarcascade_profileface.xml")
while needQuit == 0:
    foundface = 0
    #trackers.clear()
    trackers = cv2.legacy.MultiTracker_create()
    while foundface == 0:
        success, img = video.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Getting corners around the face
        #faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
        faces = faceCascade.detectMultiScale(imgGray, 1.05, 5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)  # 1.3 = scale factor, 5 = minimum neighbor
        # drawing bounding box around face
        for (x, y, w, h) in faces:
            bbox = (x, y, w, h)
            foundface = 1
            tracker = cv2.legacy.TrackerMIL_create()
            ok = tracker.init(img, bbox)
            trackers.add(tracker, img, bbox)
            #print (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        #print ("end of frame")
            #img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #bbox = (287, 23, 86, 320)
    #bbox = cv2.selectROI(frame, False)

    counter = 0
    
    
    while counter <=20:
        ok, frame = video.read()
        cuttingCoords = []
        cornersCoords = []
        idxs = []
        ok, bboxes = trackers.update(frame)
        counter +=1
        for bbox in bboxes:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            # All X Coords of boxes
            for i in range(int(bbox[2])): 
                cuttingCoords.append(int(bbox[0]) + i)
                corners = [int(bbox[0]),int(bbox[1]),int(bbox[0] + bbox[2]),int(bbox[1] + bbox[3])]
                cornersCoords.append(corners)
            # Detecting same X-val in list of boxes
            boxCoordsRepeats = []
            ccReplace = copy.deepcopy(cuttingCoords)
            for pt in cuttingCoords:
                cuttingCoords = copy.deepcopy(ccReplace)
                idxs = []
                suspects = []
                highestYs = []
                if (cuttingCoords.count(pt) > 1):
                    numOfDups = int(cuttingCoords.count(pt))
                    #print(cuttingCoords)
                    for dups in range(numOfDups):
                        idxs.append(cuttingCoords.index(pt)+dups)
                        cuttingCoords.remove(pt)
                    for idx in idxs:
                        suspects.append(cornersCoords[idx])
                        highestYs.append(cornersCoords[idx][1])
                    if any(suspects[0] in sl for sl in boxCoordsRepeats) == True:
                        #print("repeat")
                        break
                    else:
                        #print("no repeats")
                        #print(any(suspects in sl for sl in boxCoordsRepeats))
                        #print(suspects[0])
                        #print(cornersCoords)
                        boxCoordsRepeats.append(suspects[0])
                        #print(boxCoordsRepeats)
                    highestY = min(highestYs)
                    suspects.remove(suspects[highestYs.index(highestY)])
                    #print(suspects)
                    p1 = (suspects[0][0], suspects[0][1])
                    p2 = (suspects[0][2], suspects[0][3])
                    cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
                    #print("suspect caught")
            # list_duplicates(cutters)
            #print (cuttcuttiners)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            needQuit=1
            break
video.release()
cv2.destroyAllWindows()