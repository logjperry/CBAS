import sys
import vlc
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QGridLayout, QHBoxLayout, QFileDialog, QInputDialog, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import time
import ctypes
import yaml
import os

class CBAS_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        print("displaying the loading gui")

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
        self.resize(800, 800)
        
        QTimer.singleShot(3000, lambda: self.clear_layout(self.parent_layout))
        QTimer.singleShot(3500, self.load_initial)
    
    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
        self.update()

    def load_initial(self):
        print("displaying the main gui")
        
        layout = self.parent_layout
        self.createProject = QPushButton("Create a new project", self)
        self.openProject = QPushButton("Open an existing project", self)

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
                self.storage_folder = str(QFileDialog.getExistingDirectory(self, "Select Backup Directory"))
            else:
                self.storage_folder = self.project_path
            
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
                'storage_path':self.storage_folder,
                'model_diagram':'',
                'cameras':[]
            }

            with open(self.yaml_path, 'w+') as file:
                yaml.dump(config, file, allow_unicode=True)

            self.clear_layout(self.parent_layout)
        

    
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

    






if __name__ == '__main__':
    cbas = QApplication(sys.argv)

    window = CBAS_GUI()
    window.show()

    sys.exit(cbas.exec_())