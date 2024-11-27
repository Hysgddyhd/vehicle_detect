from inference_sdk import InferenceHTTPClient
from PIL import Image
import base64
import io
import os
import cv2

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    api_key="HHOQQa2OcXtsGddANg0W"
)
 
source_dir =  "/home/typer/Pictures/car_pics/Bicycle_11-7-2024/"
img="Bicycle_4.jpg"

result = client.run_workflow(
    workspace_name="robot-application",
    workflow_id="custom-workflow-q1t",
    images={
        "image": source_dir+img
    }
)
#extract visualization data form result 
data=result[0]["label_visualization"]

head,tail=os.path.splitext(img)
target = source_dir+"."+head + "_prediction" + tail

imgdata = base64.b64decode(data)
 # I assume you have a way of picking unique filenames
with open(target, 'wb') as f:
    f.write(imgdata)
    
image = cv2.imread(target)
cv2.imshow(head,image)
cv2.waitKey(0) #close windows only after any key press
cv2.destroyAllWindows()