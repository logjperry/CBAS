import sys
#import vlc
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QGridLayout, QHBoxLayout, QFileDialog, QInputDialog, QMessageBox
from PyQt5.QtWidgets import (QApplication, QMainWindow, QToolBar, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, QScrollArea, 
                             QSlider, QLineEdit)
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QSize, QCoreApplication
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QTransform
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import Qt, QRectF

from PyQt5.QtGui import QPainter, QPixmap, QPen, QPalette
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame

import streamprocessing



import time
import ctypes
import yaml
import os
import math
import streamprocessing

class CBAS_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        print("displaying the loading gui")


        self.resize(800, 800)
        self.setStyleSheet("background-color: white;")

        # https://icons8.com/icon/d6DO5ujKnnen/perch

        self.setWindowIcon(QIcon('assets/fish.png'))

        logo_pixmap = QPixmap('assets/CBAS_logo.png')
        self.logo_label = QLabel(self)
        self.logo_label.setPixmap(logo_pixmap)


        logo_hbox = QHBoxLayout()
        logo_hbox.addStretch()  
        logo_hbox.addWidget(self.logo_label)
        logo_hbox.addStretch()  

        self.parent_layout = QVBoxLayout()
        self.parent_layout.addStretch()  
        self.parent_layout.addLayout(logo_hbox)  
        self.parent_layout.addStretch()  

        self.setLayout(self.parent_layout)
        self.setWindowTitle('CBAS')


        # self.current = 0
        self.cameras = [{
            'url':'',
            'sl':1,
            'cx':0.5,
            'cy':0.5,
            'br':0,
            'co':1,
            'number':1,
            'chunk_size':30
        }]
        
        self.current = 0
        cam = self.cameras[0]

        self.url = cam['url']
        self.sl = cam['sl']
        self.cx = cam['cx']
        self.cy = cam['cy']
        self.cx = cam['br']
        self.cy = cam['co']


        self.mainloaded = False

        QTimer.singleShot(3000, lambda: self.load_initial())


    
    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            print(child)
            if child.widget():
                child.widget().close()
            elif child.layout():
                self.clear_layout(child.layout())
        self.update()

    def load_initial(self):
        print("displaying the main gui")

        self.clear_layout(self.parent_layout)
        self.update()
        
        layout = self.parent_layout
        self.createProject = QPushButton("Create a new project", self)
        self.createProject.setFont(QFont('Times', 15))
        self.openProject = QPushButton("Open an existing project", self)
        self.openProject.setFont(QFont('Times', 15))

        with open('styles/pushbutton.qss', 'r') as f:
            self.createProject.setStyleSheet(f.read())
        with open('styles/pushbutton.qss', 'r') as f:
            self.openProject.setStyleSheet(f.read())

        self.createProject.clicked.connect(self.newProject)
        self.openProject.clicked.connect(self.oldProject)

        horizontal_layout1 = QHBoxLayout()
        horizontal_layout2 = QHBoxLayout()

        horizontal_layout1.addStretch()
        horizontal_layout1.addWidget(self.createProject)
        horizontal_layout1.addStretch()

        horizontal_layout2.addStretch()
        horizontal_layout2.addWidget(self.openProject)
        horizontal_layout2.addStretch()

        layout.addStretch()
        layout.addLayout(horizontal_layout1)
        layout.addStretch()
        layout.addLayout(horizontal_layout2)
        layout.addStretch()

    
    def newProject(self):
        options = QFileDialog.Options()
        folder = str(QFileDialog.getExistingDirectory(self, "Select Parent Directory"))
        print('found this folder: ' + folder)

        # Open a dialog for the new project
        # Should ask for name and storage location
        success = False
        while not success:
            self.project_name = self.prompt_new_project()

            if self.project_name == None:
                break

            self.project_path = os.path.join(folder, self.project_name)
            try:
                os.mkdir(self.project_path)
                success = True
            except:
                QMessageBox.warning(None, 'Error', 'Unable to create the project directory. Enable permissions or try a different project name.')
                continue
        if self.project_name == None:
            self.clear_layout(self.parent_layout)
            self.load_initial()
        else:
            response = QMessageBox.question(None, 'Storage', 'The project directory was successfully created. Would you like to select a long term storage directory as a backup?')       
            if response == QMessageBox.Yes:
                self.storage_path = str(QFileDialog.getExistingDirectory(self, "Select Backup Directory"))
            else:
                self.storage_path = self.project_path
            
            self.yaml_path = os.path.join(self.project_path, 'config.yaml')

            # Yaml diagram:
                # Project name
                # Project location
                # Storage location
                # Model diagram
                # For each camera
                    # RTSP URL
                    # ffmpeg video filter
            config = {
                'project_name':self.project_name,
                'project_path':self.project_path,
                'storage_path':self.storage_path,
                'model_diagram':'',
                'cameras':self.cameras
            }

            with open(self.yaml_path, 'w+') as file:
                yaml.dump(config, file, allow_unicode=True)

            self.clear_layout(self.parent_layout)
            self.load_main()
        

    
    def oldProject(self):
        options = QFileDialog.Options()

        folder = str(QFileDialog.getExistingDirectory(self, "Select Project Directory"))

        config_path = os.path.join(folder, 'config.yaml')
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            self.project_name = config['project_name']
            self.project_path = config['project_path']
            self.storage_path = config['storage_path']
            self.model_diagram = config['model_diagram']
            self.cameras = config['cameras']

            self.yaml_path = config_path
        except:
            QMessageBox.warning(None, 'Error', 'Unable to open the project directory. Please select a project folder that contains a config file.')

        self.clear_layout(self.parent_layout)
        self.load_main()



    
    def load_main(self):


        self.url = self.cameras[self.current]['url']
        self.sl = self.cameras[self.current]['sl']
        self.cx = self.cameras[self.current]['cx']
        self.cy = self.cameras[self.current]['cy']
        self.br = self.cameras[self.current]['br']
        self.co = self.cameras[self.current]['co']
        self.number = self.cameras[self.current]['number']

        # generate pictures
        for c in self.cameras:
            try:
                streamprocessing.frameGen('test_videos/compilationWTM.mp4',0,1,'cam'+str(c['number']))
            except:
                continue
        

        self.mainloaded = True

        #self.clear_layout(self.parent_layout)

        self.setStyleSheet("background-color: gray;")
        self.inmain = True


        self.width = self.frameGeometry().width()
        self.height = self.frameGeometry().height()

        # Toolbar
        # toolbar = QToolBar()
        # self.addToolBar(toolbar)
        
        # cameras_btn = QPushButton("Cameras")
        # model_btn = QPushButton("Model")
        # analysis_btn = QPushButton("Analysis")

        # toolbar.addWidget(cameras_btn)
        # toolbar.addWidget(model_btn)
        # toolbar.addWidget(analysis_btn)

        # Main content
        main_layout = self.parent_layout
        self.clear_layout(main_layout)

        camera_scroll = QScrollArea()
        camera_container = QWidget()

        camera_container.setStyleSheet("background-color: darkgray;")
        camera_container_layout = QVBoxLayout()
        add_button_layout = QHBoxLayout()
        down_layout = QVBoxLayout()
        down_layout.addWidget(self.create_add_camera_widget())
        down_layout.addStretch()
        add_button_layout.addLayout(down_layout)
        add_button_layout.addStretch()

        camera_container_layout.addLayout(add_button_layout)


        for i in range(math.ceil((len(self.cameras)/4))):
            camera_layout = QHBoxLayout()
            for j in range(4):
                if j>=len(self.cameras)-i*4:
                    camera_layout.addWidget(self.hidden_camera_widget())
                else:
                    cam = self.create_camera_widget(i*4+j)
                    camera_layout.addWidget(cam)
                    if self.current==i*4+j:
                        self.cam_widget = cam
            camera_container_layout.addLayout(camera_layout)
            
        if len(self.cameras)<8:
            camera_container_layout.addStretch()
        camera_container.setLayout(camera_container_layout)
        self.camera_container_layout = camera_container_layout
        camera_scroll.setWidget(camera_container)
        camera_scroll.setWidgetResizable(True)

        # Bottom content
        bottom_layout = QVBoxLayout()
        text_layout = QHBoxLayout()
        slider_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        url = QLineEdit(str(self.cameras[self.current]['url']))
        url.setStyleSheet('border: 2px solid black')
        text_layout.addWidget(url)
        text_layout.addStretch(1)


        slider_layout.addWidget(Crop(self.sl, self.cx, self.cy, self.br, self.co, self))
        button_layout.addStretch(1)
        save_camera = QPushButton("Save Camera Settings")
        save_camera.setFont(QFont('Times', 15))
        save_camera.clicked.connect(self.save_cameras)
        button_layout.addWidget(save_camera)

        bottom_layout.addLayout(text_layout)
        bottom_layout.addLayout(slider_layout)
        bottom_layout.addLayout(button_layout)

        main_layout.addWidget(camera_scroll)
        main_layout.addLayout(bottom_layout)

    def add_camera(self):
        self.cameras.insert(0,{
            'url':'',
            'sl':1,
            'cx':0.5,
            'cy':0.5,
            'br':0,
            'co':1,
            'number':len(self.cameras)+1,
            'chunk_size':30
        })
        self.make_current(0)
        self.clear_layout(self.parent_layout)
        self.load_main()
    
    def make_current(self, index):
        self.current = index
        cam = self.cameras[index]

        self.url = cam['url']
        self.sl = cam['sl']
        self.cx = cam['cx']
        self.cy = cam['cy']
        self.br = cam['br']
        self.co = cam['co']
        self.number = cam['number']

        self.clear_layout(self.parent_layout)
        self.load_main()

    def create_camera_widget(self, index):
        return Camera(index, self)
    
    def hidden_camera_widget(self):
        label = QLabel()
        label.setStyleSheet("background-color: none; border:none;")
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        label.setFixedSize(int(width/5),int(width/5))
        return label

    def create_add_camera_widget(self):
        button = QPushButton(self)
        button.setFixedSize(50,50)
        button.clicked.connect(self.add_camera)
        button.setStyleSheet("border: none; background-color:none; color:gray;")
        button.setIcon(QIcon('assets/plus.svg'))
        button.setIconSize(QSize(50, 50))
        return button

        
    def resizeEvent(self, event):
        # Important: Propagate the event to the base class
        super().resizeEvent(event)
        self.width = self.frameGeometry().width()
        self.height = self.frameGeometry().height()


        print("here")
        if self.mainloaded:
            self.clear_layout(self.parent_layout)
            self.load_main()
        self.update()

    def save_cameras(self):
        config = {
            'project_name':self.project_name,
            'project_path':self.project_path,
            'storage_path':self.storage_path,
            'model_diagram':'',
            'cameras':self.cameras
        }

        with open(self.yaml_path, 'w+') as file:
            yaml.dump(config, file, allow_unicode=True)


    def update_crop(self):
        self.update_cameras()

    def update_cameras(self):
        self.clear_layout(self.camera_container_layout)
        add_button_layout = QHBoxLayout()
        down_layout = QVBoxLayout()
        down_layout.addWidget(self.create_add_camera_widget())
        down_layout.addStretch()
        add_button_layout.addLayout(down_layout)
        add_button_layout.addStretch()

        self.camera_container_layout.addLayout(add_button_layout)


        for i in range(math.ceil((len(self.cameras)/4))):
            camera_layout = QHBoxLayout()
            for j in range(4):
                if j>=len(self.cameras)-i*4:
                    camera_layout.addWidget(self.hidden_camera_widget())
                else:
                    cam = self.create_camera_widget(i*4+j)
                    camera_layout.addWidget(cam)
                    if self.current==i*4+j:
                        self.cam_widget = cam
            self.camera_container_layout.addLayout(camera_layout)
            
        if len(self.cameras)<8:
            self.camera_container_layout.addStretch()


        


    # USE CASE: Standardize incoming video
    # User should be able to see screenshots of the video streams with the corresponding video filter in place
    # User should be able to adjust brightness, contrast, etc to make the screenshots equivilent
    # User should be able start/stop recording at anytime for any camera

    # USE CASE: Specify Base Models + Post-Processor
    # User should be able to specify the base machine learning models for a project prior to recording
        # Each of these will simply output a timeseries of vectors
    # User should be able to manipulate the collected timeseries of vectors to create a random forest post-processor

    # USE CASE: Find optimal recording chunk for hardware
    # User should be presented with a test video that can time each step of the inference process (assuming the user wants a contained process sat<1)

    # USE CASE: Create a test set from a long-term recording
    # User should be able to construct a test set from a recording

    # USE CASE: Store videos to a network file server (NAS)
    # May be useless because the NAS can be mounted

    # USE CASE: Grading the model
    # Should provide the user with images or charts that give details about model performance

    def prompt_new_project(self):
        text, ok = QInputDialog.getText(self, 'Name your new project!', 'Enter a name:')
        if ok:
            print(f'User entered: {text}')
            return text
        else:
            print('User cancelled the input.')
            return None

    

class Crop(QWidget):
    def __init__(self, sl, cx, cy, br, co, parent):
        super().__init__()
        self.sl = sl 
        self.cx = cx
        self.cy = cy
        self.br = br 
        self.co = co 

        self.w = parent.width
        self.h = parent.height

        self.parent = parent

        self.init_ui()
    def init_ui(self):
        layout = QHBoxLayout()

        layout1 = QVBoxLayout()
        layout2 = QHBoxLayout()

        brLayout = QVBoxLayout()
        coLayout = QVBoxLayout()
        

        slSubLayout = QHBoxLayout()
        cxSubLayout = QHBoxLayout()
        cySubLayout = QHBoxLayout()


        sl = QSlider(Qt.Horizontal)
        cx = QSlider(Qt.Horizontal)
        cy = QSlider(Qt.Horizontal)
        br = QSlider(Qt.Vertical)
        co = QSlider(Qt.Vertical)

        slLabel = QLabel()
        cxLabel = QLabel()
        cyLabel = QLabel()
        brLabel = QLabel()
        coLabel = QLabel()

        slMin = QLabel()
        cxMin = QLabel()
        cyMin = QLabel()
        brMin = QLabel()
        coMin = QLabel()

        slMax = QLabel()
        cxMax = QLabel()
        cyMax = QLabel()
        brMax = QLabel()
        coMax = QLabel()

        slLabel.setFixedSize(int(self.w/4),int(self.h/40))
        cxLabel.setFixedSize(int(self.w/4),int(self.h/40))
        cyLabel.setFixedSize(int(self.w/4),int(self.h/40))
        brLabel.setFixedSize(int(self.w/40),int(self.h/10))
        coLabel.setFixedSize(int(self.w/40),int(self.h/10))


        sl.setFixedSize(int(self.w/4),int(self.h/30))
        cx.setFixedSize(int(self.w/4),int(self.h/30))
        cy.setFixedSize(int(self.w/4),int(self.h/30))
        br.setFixedSize(int(self.w/20),int(self.h/4))
        co.setFixedSize(int(self.w/20),int(self.h/4))

        #brLabel.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        #coLabel.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        
        slMin.setAlignment(Qt.AlignLeft)
        cxMin.setAlignment(Qt.AlignLeft)
        cyMin.setAlignment(Qt.AlignLeft)
        brMin.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        coMin.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        slMax.setAlignment(Qt.AlignRight)
        cxMax.setAlignment(Qt.AlignRight)
        cyMax.setAlignment(Qt.AlignRight)
        brMax.setAlignment(Qt.AlignCenter | Qt.AlignBottom)
        coMax.setAlignment(Qt.AlignCenter | Qt.AlignBottom)

        with open('styles/qslider.qss', 'r') as f:
            sl.setStyleSheet(f.read())
        with open('styles/qslider.qss', 'r') as f:
            cx.setStyleSheet(f.read())
        with open('styles/qslider.qss', 'r') as f:
            cy.setStyleSheet(f.read())
        with open('styles/qslider.qss', 'r') as f:
            br.setStyleSheet(f.read())
        with open('styles/qslider.qss', 'r') as f:
            co.setStyleSheet(f.read())


        sl.setRange(0,100)
        cx.setRange(0,100)
        cy.setRange(0,100)
        br.setRange(-100,100)
        co.setRange(0,200)

        sl.setValue(int(self.sl*100))
        cx.setValue(int(self.cx*100))
        cy.setValue(int(self.cy*100))
        br.setValue(int(self.br*100))
        co.setValue(int(self.co*100))

        slLabel.setText('Scale')
        cxLabel.setText('Crop X')
        cyLabel.setText('Crop Y')
        brLabel.setText('Bright')
        coLabel.setText('Con')

        slMin.setText(str(sl.minimum()))
        cxMin.setText(str(cx.minimum()))
        cyMin.setText(str(cy.minimum()))
        brMin.setText('-1')
        coMin.setText('0')

        slMax.setText(str(sl.maximum()))
        cxMax.setText(str(cx.maximum()))
        cyMax.setText(str(cy.maximum()))
        brMax.setText('1')
        coMax.setText('2')

        sl.valueChanged.connect(lambda:self.setSL(sl.value()))
        cx.valueChanged.connect(lambda:self.setCX(cx.value()))
        cy.valueChanged.connect(lambda:self.setCY(cy.value()))
        br.valueChanged.connect(lambda:self.setBR(br.value()/100))
        co.valueChanged.connect(lambda:self.setCO(co.value()/100))
        
        layout1.addWidget(slLabel)
        layout1.addWidget(sl)
        slSubLayout.addWidget(slMin)
        slSubLayout.addWidget(slMax)
        layout1.addLayout(slSubLayout)

        layout1.addWidget(cxLabel)
        layout1.addWidget(cx)
        cxSubLayout.addWidget(cxMin)
        cxSubLayout.addWidget(cxMax)
        layout1.addLayout(cxSubLayout)

        layout1.addWidget(cyLabel)
        layout1.addWidget(cy)
        cySubLayout.addWidget(cyMin)
        cySubLayout.addWidget(cyMax)
        layout1.addLayout(cySubLayout)
        
        layout1.addStretch()

        brLayout.addWidget(brMax)
        brLayout.addWidget(br)
        brLayout.addWidget(brMin)
        brLayout.addWidget(brLabel)
        brLayout.addStretch()

        
        coLayout.addWidget(coMax)
        coLayout.addWidget(co)
        coLayout.addWidget(coMin)
        coLayout.addWidget(coLabel)
        coLayout.addStretch()

        layout2.addLayout(brLayout)
        layout2.addLayout(coLayout)
        layout2.addStretch()

        layout.addLayout(layout1)
        layout.addLayout(layout2)


        parent_layout = QHBoxLayout()
        parent_layout.addLayout(layout)
        parent_layout.addStretch()

        self.setLayout(parent_layout)

    def setSL(self, val):
        self.parent.cameras[self.parent.current]['sl'] = val/100
        self.parent.update_crop()
    def setCX(self, val):
        self.parent.cameras[self.parent.current]['cx'] = val/100
        self.parent.update_crop()
    def setCY(self, val):
        self.parent.cameras[self.parent.current]['cy'] = val/100
        self.parent.update_crop()
    def setBR(self, val):
        self.parent.cameras[self.parent.current]['br'] = val/100
        self.parent.update_crop()
    def setCO(self, val):
        self.parent.cameras[self.parent.current]['co'] = val/100
        self.parent.update_crop()
        


class Camera(QWidget):
    clicked = pyqtSignal()

    def __init__(self, index, parent):
        super().__init__()
        self.index = index

        self.width = parent.width
        self.height = parent.height

        self.cam = parent.cameras[index]

        self.sl = self.cam['sl']
        self.cx = (self.cam['cx'] -.5)*2
        self.cy = (self.cam['cy'] -.5)*2
        

        self.parent = parent


        self.init_ui()

    def resizeEvent(self, event):
        # This ensures the image rescales with the window resize
        self.updatePixmap()
        super().resizeEvent(event)

    def updatePixmap(self):
        # Scale pixmap to fit the current size and keep its aspect ratio
        scaled_pixmap = self.pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatioByExpanding)
        self.imageLabel.setPixmap(scaled_pixmap)

    
    def init_ui(self):


        layout = QHBoxLayout()


        # Create a label that will contain the image
        self.imageLabel = QLabel(self)
        self.imageLabel.setStyleSheet('border:none;padding:0px; margin:0px;')
        
        # Load the image
        self.pixmap = QPixmap('frames/cam'+str(self.parent.cameras[self.index]['number'])+'.png')
    
        if self.pixmap.isNull():
            self.pixmap = QPixmap('assets/noimage.png')



        # Scale the pixmap to fit the current size and keep its aspect ratio
        scaled_pixmap = self.pixmap.scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatioByExpanding)

        # Set the pixmap to the label
        self.imageLabel.setPixmap(scaled_pixmap)

        #self.setStyleSheet("background-image: url('frames/cam1.jpg'); background-position: center; background-repeat: no-repeat; background-size: cover;")
       
        isl = (1 - self.sl)/2




        label = BorderedLabel()
        self.imageLabel.setStyleSheet('border:none; padding:0px; margin:0px;')
        # Set transparent background and ensure no system background is drawn
        label.setAttribute(Qt.WA_NoSystemBackground)
        label.setAttribute(Qt.WA_TranslucentBackground)

        if self.index != self.parent.current:
            label.setStyleSheet("border: 5px solid red;")
        else:
            label.setStyleSheet("border: 5px solid red;")


        w = self.frameGeometry().width()
        h = self.frameGeometry().height()

        size = int(self.width/5)
        self.imageLabel.setFixedSize(size,size)


        left = int(size*isl)
        right = int(size*isl)
        top = int(size*isl)
        bot = int(size*isl)
        if self.cx>0:
            left = int(left + self.cx*size/2)
            right = int(right - self.cx*size/2)
            if right<0:
                right = 0
        else:
            left = int(left + self.cx*size/2)
            right = int(right - self.cx*size/2)
            if left<0:
                left = 0
        if self.cy>0:
            top = int(top + self.cy*size/2)
            bot = int(bot - self.cy*size/2)
            if bot<0:
                bot = 0
        else:
            top = int(top + self.cy*size/2)
            bot = int(bot - self.cy*size/2)
            if top<0:
                top = 0
        layout.setContentsMargins(left, top, right, bot)


        self.clicked.connect(lambda: self.parent.make_current(self.index))

        layout.addWidget(label)

        self.imageLabel.setLayout(layout)

        self.parentlayout = QHBoxLayout()
        self.parentlayout.addWidget(self.imageLabel)

        self.setLayout(self.parentlayout)
    

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # Emit the clicked signal
        self.clicked.emit()

    def update_crop(self, sl, cx, cy):
        self.sl = sl
        self.cx = cx 
        self.cy = cy

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().delete()
            elif child.layout():
                self.clear_layout(child.layout())
        self.update()

    def draw(self):

        layout = QHBoxLayout()

        image = QWidget()

        image.setStyleSheet('border:none; background-color:white; padding:0px; margin:0px;')

        self.setStyleSheet("border: none; background-color:white;")

        isl = (1 - self.sl)/2

        label = QLabel()
        if self.index!=self.parent.current:
            label.setStyleSheet("border: none; background-color:none;")
        else:
            label.setStyleSheet("border: none; background-color:none;")


        w = self.frameGeometry().width()
        h = self.frameGeometry().height()

        size = int(self.width/5)
        image.setFixedSize(size,size)


        layout.setContentsMargins(int(size*isl), int(size*isl), int(size*isl), int(size*isl))


        self.clicked.connect(lambda: self.parent.make_current(self.index))

        layout.addWidget(label)

        image.setLayout(layout)

        self.parentlayout.addWidget(image)


class BorderedLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super(BorderedLabel, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_TranslucentBackground)  # Enable transparent background

    def paintEvent(self, event):
        # Start by filling the background with a transparent color
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.transparent)  # Fill the rect with a transparent color
        
        # Set up the pen with the color and width of the border
        pen = QPen(Qt.red)
        pen.setWidth(8)  # Change the width to the desired border width
        painter.setPen(pen)

        # Draw a rectangle on the very edge of the widget
        painter.drawRect(self.rect().adjusted(0, 0, -pen.width(), -pen.width()))

        super(BorderedLabel, self).paintEvent(event)

if __name__ == '__main__':
    cbas = QApplication(sys.argv)


    window = CBAS_GUI()
    window.show()



    sys.exit(cbas.exec_())