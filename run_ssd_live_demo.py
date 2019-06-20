from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys

#################################################
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVision import DroneVision
import threading
import cv2
import time
import numpy as np
########################################################

STATE_SEARCH = 'Search'
STATE_ALIGNMENT = 'Alignment'
STATE_MOVING = 'Moving'
STATE_LANDING = 'Landing'

if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

'''
if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(1)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)
'''
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)


timer = Timer()



########################################################################################
class UserVision:

    def __init__(self, vision):
        #self.index = 0
        self.vision = vision
        self.target = False
        self.targetPoint = np.array([0, 0])
        self.sourceImageCenterPoint = np.array([0, 0])
        self.targetArea = 0
        self.frameArea = 0
        self.alignmentTolerance = 90
        self.state = STATE_SEARCH
        self.terminar = False

    # Funcion para obtener el objetivo de aterrizaje
    def get_target(self):
        return self.target

    def save_pictures(self, args):
        # print("in save pictures on image %d " % self.index)

        orig_image = self.vision.get_latest_valid_picture()
        # self.sourceImageCenterPoint = np.array([orig_image.shape[0]/2, orig_image.shape[1]/2])
        self.sourceImageCenterPoint = np.array([320, 180])
        # print(orig_image.shape)

        self.frameArea = orig_image.shape[0]*orig_image.shape[1]

        if (orig_image is not None):

            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            timer.start()
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
            interval = timer.end()
            #print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
                cv2.putText(orig_image, label,
                            (box[0] + 20, box[1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            (255, 0, 255),
                            2)  # line type



                # labels : 9:silla  ,20: monitor    ; 8: gato
                if (labels[i].data.cpu().numpy() == 15):
                    # print(box.data.cpu().numpy())
                    # print(box)
                    numpyBox = box.data.cpu().numpy()
                    medio_x = (box[2] - box[0]) / 2
                    medio_y = (box[3] - box[1]) / 2
                    puntoMedio = np.array([medio_x + box[0], medio_y + box[1]])

                    self.targetPoint = puntoMedio
                    area = np.array(np.abs((box[0] - box[2]) * (box[3] - box[1])))
                    self.targetArea = area
                    self.target = True
                    print(puntoMedio)
                    print('Area objetivo: '+ str(area))
                    if self.targetArea >= (360*640*0.4):
                        self.terminar = True
                else:
                    self.target = False

            cv2.circle(orig_image, (int(userVision.sourceImageCenterPoint[0]), int(userVision.sourceImageCenterPoint[1])), userVision.alignmentTolerance, (0, 255, 120), 3)

            cv2.putText(orig_image, 'State: ' + self.state,
                        (0, 340),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (0, 255, 0),
                        2)  # line type

            if self.target:
                cv2.circle(orig_image, (self.targetPoint[0], self.targetPoint[1]), 5, (0, 0, 255), -2)

            cv2.imshow('Imagen con objetos detectados', orig_image)

            cv2.waitKey(1)
            #self.index += 1
            # print(self.index)


mamboAddr = "e0:14:d0:63:3d:d0"

def rotationalSearch():
    userVision.state = STATE_SEARCH
    if not userVision.target:
        mambo.fly_direct(roll=0, pitch=0, yaw=-15, vertical_movement=0, duration=1)
    while (not userVision.target):
        mambo.smart_sleep(2)
        print('Estado de busqueda')
        print('Objetivo Encontrado: {}', userVision.target)
        print('Ubicacion objetivo de aterrizaje: {}'.format(userVision.targetPoint))
        # print('Ubicacion centro de la Imagen: {}'.format(userVision.sourceImageCenterPoint))
        mambo.fly_direct(roll=0, pitch=0, yaw=15, vertical_movement=0, duration=1)



def alinearTarget():
    while (userVision.target):
        print('Alineando Objetivo')
        userVision.state = STATE_ALIGNMENT
        distanciaX = userVision.sourceImageCenterPoint[0] - userVision.targetPoint[0]
        print("distancia en x es {}".format(distanciaX))
        distanciaY = userVision.sourceImageCenterPoint[1] - userVision.targetPoint[1]
        velocity_multiplier_x = 0
        velocity_multiplier_y = 0
        if np.abs(distanciaX) > userVision.alignmentTolerance:
            velocity_multiplier_x = -distanciaX / 320
        if np.abs(distanciaY) > userVision.alignmentTolerance:
            velocity_multiplier_y = distanciaY / 180
        print("Multiplicadores x e y: {}".format((velocity_multiplier_x, velocity_multiplier_y)))
        yaw = 30 * velocity_multiplier_x
        verticalMovement = 20 * velocity_multiplier_y
        print("Yaw, vertical: {}".format((yaw, verticalMovement)))
        if yaw == 0 and verticalMovement == 0:
          break
        mambo.fly_direct(roll=0, pitch=0, yaw=int(yaw), vertical_movement=int(verticalMovement), duration=1)
        mambo.smart_sleep(2)

def avanzar():
    print('Avanzando hacia el objetivo')
    userVision.state = STATE_MOVING
    mambo.fly_direct(roll=0, pitch=15, yaw=0, vertical_movement=0, duration=1)
    mambo.smart_sleep(2)

mambo = Mambo(mamboAddr, use_wifi=True)
print("trying to connect to mambo now")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

if (success):
    # get the state information
    print("sleeping")
    mambo.smart_sleep(1)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)

    print("Preparing to open vision")
    mamboVision = DroneVision(mambo, is_bebop=False, buffer_size=30)
    userVision = UserVision(mamboVision)
    mamboVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
    success = mamboVision.open_video()
    print("Success in opening vision is %s" % success)

    if (success):
        print("Vision successfully started!")
        # get the state information
        print("sleeping")
        mambo.smart_sleep(2)
        mambo.ask_for_state_update()
        mambo.smart_sleep(2)

        print("taking off!")
        mambo.safe_takeoff(5)

        if (mambo.sensors.flying_state != "emergency"):
            print("flying state is %s" % mambo.sensors.flying_state)
            print("Flying direct: going up")
            mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=1)

            terminar = False

            #rotationalSearch()

            while not userVision.terminar:
                rotationalSearch()
                while userVision.target and not userVision.terminar:
                    alinearTarget()
                    if not userVision.target:
                        break
                    avanzar()


            '''
            print("flying state is %s" % mambo.sensors.flying_state)
            print("Flying direct: going up")
            mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=1)

            print("flip left")
            print("flying state is %s" % mambo.sensors.flying_state)
            success = mambo.flip(direction="left")
            print("mambo flip result %s" % success)
            mambo.smart_sleep(5)

            print("flip right")
            print("flying state is %s" % mambo.sensors.flying_state)
            success = mambo.flip(direction="right")
            print("mambo flip result %s" % success)
            mambo.smart_sleep(5)

            print("flip front")
            print("flying state is %s" % mambo.sensors.flying_state)
            success = mambo.flip(direction="front")
            print("mambo flip result %s" % success)
            mambo.smart_sleep(5)

            print("flip back")
            print("flying state is %s" % mambo.sensors.flying_state)
            success = mambo.flip(direction="back")
            print("mambo flip result %s" % success)
            '''
            mambo.smart_sleep(1)

            print("landing")
            userVision.state = STATE_LANDING
            print("flying state is %s" % mambo.sensors.flying_state)
            mambo.safe_land(5)
            mambo.smart_sleep(1)

        print("Ending the sleep and vision")
        cv2.destroyAllWindows()
        mamboVision.close_video()
        mambo.smart_sleep(5)



    print("disconnecting")
    mambo.disconnect()


##########################################################################################

