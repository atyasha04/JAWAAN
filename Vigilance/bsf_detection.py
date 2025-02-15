"""import cv2
import os
import numpy as np
import statistics
from matplotlib import pyplot as plt
from ultralytics import YOLO

x_mouse = 0
y_mouse = 0

# Initialize YOLO model with custom weights
model = YOLO("final_jawaan_final_120.pt")

# Initialize video capture
vid = cv2.VideoCapture('test_video1.mp4')  # Webcam
# Uncomment the desired input source:
# vid = cv2.VideoCapture("path_to_video.mp4")  

height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = vid.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("mytest_gray.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height), isColor=False)
out2 = cv2.VideoWriter("mytest_RGB.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

def mouse_events(event, x, y, flags, param):
   
    global x_mouse, y_mouse
    if event == cv2.EVENT_MOUSEMOVE:
        x_mouse, y_mouse = x, y

# Dictionary to log counts
D1 = {}
num = 0
L2 = []  # Stores the count of Unauthorized in batches
L3 = []  # Stores the count of BSF in batches

while True:
    L1 = []  # List to collect detected objects
    num += 1
    success, frame = vid.read()
    if not success:
        break
    
    frame_resized = cv2.resize(frame, (960, 540))
    
    # Object detection
    results = model.predict(frame_resized)
    result = results[0]
    
    bsf_count = 0
    unauthorized_count = 0

    # Process detected boxes
    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        if label == "Unauthorized":
            unauthorized_count += 1
        elif label == "BSF":
            bsf_count += 1
    
    # Add counts to lists
    if num % 10 != 0:
        L2.append(unauthorized_count)
        L3.append(bsf_count)
    else:
        D1[num] = {"Unauthorized": max(L2), "BSF": max(L3)}
        L2 = []
        L3 = []

    # Temperature tracking
    gray8 = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
    data = gray8[y_mouse, x_mouse]
    cv2.circle(gray8, (x_mouse, y_mouse), 2, (0, 0, 0), -1)
    cv2.putText(gray8, f"{data:.1f}", (x_mouse - 80, y_mouse - 15), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)

    


    result_temp= cv2.applyColorMap(gray8, cv2.COLORMAP_JET)
    result_temp2= cv2.applyColorMap(gray8, cv2.COLORMAP_MAGMA)
    screen1= np.concatenate((result_temp,frame_resized), axis=0)
    screen2= np.concatenate((result.plot(),result_temp2), axis=0)
    screen=np.concatenate((screen1,screen2),axis=1)

    # Display counts
    #cv2.putText(screen, f"Unauthorized: {unauthorized_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #cv2.putText(screen, f"BSF: {bsf_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", screen)
    out.write(gray8)
    out2.write(frame)

    cv2.setMouseCallback("Output", mouse_events)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# Release resources
vid.release()
out.release()
out2.release()
cv2.destroyAllWindows()

# Process results
unauthorized_counts = [v["Unauthorized"] for v in D1.values()]
bsf_counts = [v["BSF"] for v in D1.values()]
frames = list(D1.keys())
mean_unauthorized = statistics.mean(unauthorized_counts)

# Print results
print("Frame Stats:", D1)
print("Mean Unauthorized:", mean_unauthorized)
for frame, counts in D1.items():
    if counts["Unauthorized"] >= mean_unauthorized:
        print(f"Frame {frame}: Unauthorized count is high.")
    else:
        print(f"Frame {frame}: Unauthorized count is low.")"""

"""# Plot results
plt.figure(figsize=(10, 5))
plt.bar([str(f) for f in frames], unauthorized_counts, label="Unauthorized", alpha=0.7, color='r')
plt.bar([str(f) for f in frames], bsf_counts, label="BSF", alpha=0.7, color='g')
plt.title("Detection Counts")
plt.ylabel("Count")
plt.xlabel("Frame Number")
plt.legend()
plt.show()"""




"""import cv2
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Function to send an emergency alert
def send_alert():
    sender_email = "your_email@example.com"
    sender_password = "your_password"
    recipient_email = "base_camp_email@example.com"
    subject = "Emergency Alert: Unauthorized Person Detected"
    body = "An unauthorized person has been detected. Please take immediate action."

    # Set up the email message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Send the email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        print("Alert sent successfully.")
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Thermal mapping
def rgb_to_thermal(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thermal_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    return thermal_image

# Magma mapping
def rgb_to_magma(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    magma_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_MAGMA)
    return magma_image

# Object detection
def detect_objects(frame, model):
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    unauthorized_detected = False
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        label = "BSF" if int(class_id) == 0 else "Unauthorized"

        # Draw bounding box and label
        color = (0, 255, 0) if label == "BSF" else (0, 0, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check for unauthorized person
        if label == "Unauthorized":
            unauthorized_detected = True

    # Send alert if unauthorized person is detected
    if unauthorized_detected:
        send_alert()

    return frame

# Main execution
def main():
    # Load the fine-tuned YOLO model
    model = YOLO("jawaan1.pt")  # Replace with your model file path

    # Load video feed (0 for webcam or provide video file path)
    cap = cv2.VideoCapture(0)

    # Get screen resolution for full-screen display
    screen_width = 1920  # Set your screen width
    screen_height = 1080  # Set your screen height

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Thermal and magma mapping
        thermal_frame = rgb_to_thermal(frame)
        magma_frame = rgb_to_magma(frame)

        # Perform object detection
        detection_frame = detect_objects(frame.copy(), model)

        # Resize frames for 2x2 grid
        grid_width = screen_width // 2
        grid_height = screen_height // 2

        thermal_frame = cv2.resize(thermal_frame, (grid_width, grid_height))
        magma_frame = cv2.resize(magma_frame, (grid_width, grid_height))
        detection_frame = cv2.resize(detection_frame, (grid_width, grid_height))
        original_frame = cv2.resize(frame, (grid_width, grid_height))

        # Combine frames into a 2x2 grid
        top_row = np.hstack((original_frame, thermal_frame))
        bottom_row = np.hstack((detection_frame, magma_frame))
        combined_frame = np.vstack((top_row, bottom_row))

        # Display the combined frame in full-screen
        cv2.namedWindow("Border Monitoring", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Border Monitoring", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Border Monitoring", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()"""








"""import cv2
import os
import numpy as np
import seaborn as sb
import pandas as pd
from matplotlib import pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from ultralytics import YOLO
import statistics

x_mouse=0
y_mouse=0


plot1 = plt.subplot2grid((2, 2), (0, 0))
plot2 = plt.subplot2grid((2, 2), (1, 0), rowspan=2)

model = YOLO("jawaan1.pt") 


vid=cv2.VideoCapture(0) #webcam

#vid=cv2.VideoCapture("https://videos.pexels.com/video-files/1462474/1462474-hd_1080_1906_25fps.mp4")    #not working
#vid=cv2.VideoCapture("http://192.168.174.200:81/stream") #working
#vid=cv2.VideoCapture("final_test_case2.mp4")

height =int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width =int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
fps=vid.get(cv2.CAP_PROP_FPS)
out=cv2.VideoWriter(("mytest.mp4"),cv2.VideoWriter_fourcc('P','I','M','1'),fps,(width,height),isColor=False)
out2=cv2.VideoWriter(("mytest_RGB.mp4"),cv2.VideoWriter_fourcc('P','I','M','1'),fps,(width,height),isColor=False)


def mouse_events(event,x,y,flags,param):

    if event==cv2.EVENT_MOUSEMOVE:

        global x_mouse
        global y_mouse

        x_mouse=x
        y_mouse=y

D1={} # dictionary to add details
num=0
L2=[]

while True:
    L1=[]   # list to collect the data of objects detected 
    num=num+1
    print(num)
    success,frame=vid.read()
    print("Start")
    frame2= cv2.resize(frame,(960,540))
    
    # object detection
    results = model.predict(frame2)
    
    result=results[0]
    #cv2.imshow("frame",result.plot())
    for box in result.boxes:
        label=result.names[box.cls[0].item()]
        if label=="Unauthorized":
            L1.append(label)
        #print("object",label)
    no_of_per=len(L1)
    if num%10!=0:
        L2.append(no_of_per)
    else:
        
        print(L2)
        print(max(L2))
        D1[num]=max(L2)
        L2=[]

    # temperature part
    
    gray8=cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    gray82=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    data=gray8[y_mouse,x_mouse]
    print(data)

    cv2.circle(gray8, (x_mouse, y_mouse), 2, (0, 0, 0), -1)
    cv2.putText(gray8,"{0:.1f}".format(data),(x_mouse-80, y_mouse-15), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5)

    result_temp= cv2.applyColorMap(gray8, cv2.COLORMAP_JET)
    result_temp2= cv2.applyColorMap(gray8, cv2.COLORMAP_MAGMA)
    screen1= np.concatenate((result_temp,frame2), axis=0)
    screen2= np.concatenate((result.plot(),result_temp2), axis=0)
    screen=np.concatenate((screen1,screen2),axis=1)

    cv2.imshow("out",screen)
    out.write(gray82)
    out2.write(frame)
    
    cv2.setMouseCallback("out",mouse_events)


    if cv2.waitKey(10) & 0xFF==ord("q"):
        break
        
cv2.destroyAllWindows()
out.release()
print(D1)
print(D1.values())
key_lst=list(D1.keys())
key_values=list(D1.values())
print("Frames: ",key_lst)
key_new=[]
compare=[]
for i in key_lst:
    key_new.append(str(i))
    compare.append(i)
print("frames(str): ",key_new)
med=statistics.mean(key_values)

print("Mean value of Unauthorized person: ",med)
for i in compare:
    if D1[i] >= med:
        print(f"area {i} is high")
    elif D1[i]< med:
        print(f"area {i} is low")
value_lst=list(D1.values())
print(value_lst)

#print("HI")


plt.figure(figsize=(10,5))
plt.bar(key_new,value_lst)
#plt.plot(key_new,value_lst,marker='o')
plt.title('max Unauthorized person')
plt.ylabel('Unauthorized person')
plt.xlabel('no of frames')

plt.show()
#plt.savefig("output.jpg")"""

import cv2
import os
import numpy as np
import statistics
from ultralytics import YOLO
from twilio.rest import Client

# Twilio credentials
TWILIO_ACCOUNT_SID = "ACdd58bf75d66b80b1afb18555314313c5"
TWILIO_AUTH_TOKEN = "7d7f058f11a60131b0b42b9c707e1ae1"
TWILIO_PHONE_NUMBER = "+12316743158"  # Twilio number
ALERT_PHONE_NUMBER = "+919474716149"  # Verified recipient's number

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

x_mouse = 0
y_mouse = 0

# Initialize YOLO model with custom weights
model = YOLO("final_jawaan_final_120.pt")

# Initialize video capture
vid = cv2.VideoCapture('test_video1.mp4')
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = vid.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("mytest_gray.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height), isColor=False)
out2 = cv2.VideoWriter("mytest_RGB.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

def send_sms_alert(frame_number):
    """Send an SMS alert for unauthorized detection."""
    try:
        message = f"ALERT: Unauthorized person detected in frame {frame_number}"
        response = twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print(f"SMS Alert Sent: {message}, SID: {response.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

def mouse_events(event, x, y, flags, param):
    """Track mouse movement."""
    global x_mouse, y_mouse
    if event == cv2.EVENT_MOUSEMOVE:
        x_mouse, y_mouse = x, y

# Dictionary to log counts
D1 = {}
num = 0
L2 = []  # Stores the count of Unauthorized in batches
L3 = []  # Stores the count of BSF in batches

while True:
    num += 1
    success, frame = vid.read()
    if not success:
        break

    frame_resized = cv2.resize(frame, (960, 540))
    
    # Object detection
    results = model.predict(frame_resized)
    result = results[0]
    
    bsf_count = 0
    unauthorized_count = 0

    # Process detected boxes
    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        confidence = box.conf[0].item() * 100  # Confidence in percentage

        if label == 1:
            unauthorized_count += 1
            # Trigger SMS alert if confidence > 30%
            #if confidence > 30:
            send_sms_alert(num)
        elif label == "BSF":
            bsf_count += 1

    # Add counts to lists
    if num % 10 != 0:
        L2.append(unauthorized_count)
        L3.append(bsf_count)
    else:
        D1[num] = {"Unauthorized": max(L2), "BSF": max(L3)}
        L2 = []
        L3 = []

    # Temperature tracking
    gray8 = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
    data = gray8[y_mouse, x_mouse]
    cv2.circle(gray8, (x_mouse, y_mouse), 2, (0, 0, 0), -1)
    cv2.putText(gray8, f"{data:.1f}", (x_mouse - 80, y_mouse - 15), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)

    result_temp = cv2.applyColorMap(gray8, cv2.COLORMAP_JET)
    result_temp2 = cv2.applyColorMap(gray8, cv2.COLORMAP_MAGMA)
    screen1 = np.concatenate((result_temp, frame_resized), axis=0)
    screen2 = np.concatenate((result.plot(), result_temp2), axis=0)
    screen = np.concatenate((screen1, screen2), axis=1)

    # Display counts
    cv2.imshow("Output", screen)
    out.write(gray8)
    out2.write(frame)

    cv2.setMouseCallback("Output", mouse_events)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# Release resources
vid.release()
out.release()
out2.release()
cv2.destroyAllWindows()

# Process results
unauthorized_counts = [v["Unauthorized"] for v in D1.values()]
bsf_counts = [v["BSF"] for v in D1.values()]
frames = list(D1.keys())
mean_unauthorized = statistics.mean(unauthorized_counts)

# Print results
print("Frame Stats:", D1)
print("Mean Unauthorized:", mean_unauthorized)
for frame, counts in D1.items():
    if counts["Unauthorized"] >= mean_unauthorized:
        print(f"Frame {frame}: Unauthorized count is high.")
    else:
        print(f"Frame {frame}: Unauthorized count is low.")
















"""import cv2
import os
import numpy as np
from twilio.rest import Client
from ultralytics import YOLO

# Twilio configuration
ACCOUNT_SID = "ACdd58bf75d66b80b1afb18555314313c5"
AUTH_TOKEN = "7d7f058f11a60131b0b42b9c707e1ae1"
TWILIO_PHONE = "+12316743158"
TO_PHONE = "9474716149"

# Function to send SMS alert
def send_sms_alert():
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            body="Emergency Alert: An unauthorized person has been detected with high probability. Immediate action is required.",
            from_=TWILIO_PHONE,
            to=TO_PHONE
        )
        print("SMS alert sent successfully. SID:", message.sid)
    except Exception as e:
        print(f"Failed to send SMS alert: {e}")

# Initialize YOLO model with custom weights
model = YOLO("final_jawaan_final_120.pt")

# Initialize video capture
vid = cv2.VideoCapture('test_video1.mp4')  # Replace with your video file path
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = vid.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("mytest_gray.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height), isColor=False)
out2 = cv2.VideoWriter("mytest_RGB.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

while True:
    success, frame = vid.read()
    if not success:
        break

    frame_resized = cv2.resize(frame, (960, 540))

    # Object detection
    results = model.predict(frame_resized)
    result = results[0]

    unauthorized_detected = False

    for box in result.boxes:
        confidence = box.conf.item() * 100  # Confidence as percentage
        label = result.names[int(box.cls.item())]
        
        if label == "Unauthorized":
            unauthorized_detected = True
            print(f"Unauthorized person detected with confidence: {confidence:.2f}%")
            send_sms_alert()
            break  # Send alert for the first unauthorized detection in the frame

    # Temperature tracking (Optional)
    gray8 = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)

    # Display results
    result_temp = cv2.applyColorMap(gray8, cv2.COLORMAP_JET)
    result_temp2 = cv2.applyColorMap(gray8, cv2.COLORMAP_MAGMA)
    screen1 = np.concatenate((result_temp, frame_resized), axis=0)
    screen2 = np.concatenate((result.plot(), result_temp2), axis=0)
    screen = np.concatenate((screen1, screen2), axis=1)

    cv2.imshow("Output", screen)
    out.write(gray8)
    out2.write(frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# Release resources
vid.release()
out.release()
out2.release()
cv2.destroyAllWindows()"""
