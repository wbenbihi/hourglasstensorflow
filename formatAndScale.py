# This function does 2 things:

# 1) Changes from List to Dict, so you can access the coordinates using the name of a body part, i.e. "Right Ankle"

# 2) Scales the data from 256x256 back to your original height and width
# For example, if original image is 1280x720, the returned result will be the X,Y coordinates for a 1280x720 image, rather than for a 256x256 image

# Takes the prediction from hourglass, plus the original height and width, as input. Returns a dict of coordinates.

def formatAndScaleCoords(predictions,height,width):

    result = {}

    for i, coord in enumerate(predictions):
        y_original = int(coord[0])
        x_original = int(coord[1])
		
        y_percentage = y_original / 256
        x_percentage = x_original / 256

        y_scaled = int(y_percentage * height)
        x_scaled = int(x_percentage * width)

        if i == 0:

            result["Right Ankle"] = (x_scaled,y_scaled)

        elif i == 1:

            result["Right Knee"] = (x_scaled,y_scaled)

        elif i == 2:

            result["Right Hip"] = (x_scaled,y_scaled)

        elif i == 3:

            result["Left Hip"] = (x_scaled,y_scaled)

        elif i == 4:

            result["Left Knee"] = (x_scaled,y_scaled)

        elif i == 5:

            result["Left Ankle"] = (x_scaled,y_scaled)

        elif i == 6:

            result["Pelvis"] = (x_scaled,y_scaled)

        elif i == 7:

            result["Thorax"] = (x_scaled,y_scaled)

        elif i == 8:

            result["Neck"] = (x_scaled,y_scaled)

        elif i == 9:

            result["Head"] = (x_scaled,y_scaled)

        elif i == 10:

            result["Right Wrist"] = (x_scaled,y_scaled)

        elif i == 11:

            result["Right Elbow"] = (x_scaled,y_scaled)

        elif i == 12:

            result["Right Shoulder"] = (x_scaled,y_scaled)

        elif i == 13:

            result["Left Shoulder"] = (x_scaled,y_scaled)

        elif i == 14:

            result["Left Elbow"] = (x_scaled,y_scaled)

        elif i == 15:

            result["Left Wrist"] = (x_scaled,y_scaled)

    return result
