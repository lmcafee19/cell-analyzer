# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50000):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.object_area = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    # TODO Create version of update using circles
    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        centroid_rect_dict = OrderedDict()
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

            # Create mapping between rects and centroids
            centroid_rect_dict[(cX, cY)] = [startX, startY, endX, endY]


        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.object_area[self.nextObjectID] = calc_rect_area(centroid_rect_dict[tuple(inputCentroids[i])])
                self.register(inputCentroids[i])

        # otherwise, we are currently tracking objects, so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids, "euclidean")

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # Record data mapped between this cell id and rectangle boundries
                self.object_area[objectID] = calc_rect_area(centroid_rect_dict[tuple(inputCentroids[col])])

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

                # compute both the row and column index we have NOT yet
                # examined
                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                # in the event that the number of object centroids is
                # equal or greater than the number of input centroids
                # we need to check and see if some of these objects have
                # potentially disappeared
                if D.shape[0] >= D.shape[1]:
                    # loop over the unused row indexes
                    for row in unusedRows:
                        # grab the object ID for the corresponding row
                        # index and increment the disappeared counter
                        objectID = objectIDs[row]
                        # TODO add handling for after an object disappears as to not leave holes in the disappeared dict keys
                        self.disappeared[objectID] += 1
                        # check to see if the number of consecutive
                        # frames the object has been marked "disappeared"
                        # for warrants deregistering the object
                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)

            # After all known centroids have been updated to track their new position, register all newly found centroids
            # If there are more centroids input than tracked
            if D.shape[1] > D.shape[0]:
                # Convert np arrays to lists
                input = list(inputCentroids)
                known = list(self.objects.values())

                # Create set of new centroids
                # since sets can only contain unique elements all known centroids will be removed
                new_centroids = set(tuple(i) for i in input)
                known_centroids = set(tuple(k) for k in known)
                unique_centroids = new_centroids - known_centroids
                #print(f"New: {len(new_centroids)}, Known: {len(known_centroids)}, Unique: {len(unique_centroids)}")

                # Register all unique centroids
                for centroid in unique_centroids:
                    # Map new id to new rectangle bounds
                    self.object_area[self.nextObjectID] = calc_rect_area(centroid_rect_dict[centroid])
                    self.register(centroid)



        # return the set of trackable objects
        # TODO Possibly combine dicts
        return self.objects, self.object_area

'''
    Calculates the area of the given rectangle
    @param rectangle: List Containing starting x coordinate, starting y, ending x, and ending y in that order
    @return The area of the given rectangle
'''
def calc_rect_area(rectangle):
    # Grab length and Width
    length = rectangle[3] - rectangle[1]
    width = rectangle[2] - rectangle[0]
    # A = l * w
    area = length * width
    return area