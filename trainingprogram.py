import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QFileDialog, QLabel, QGraphicsView, QGraphicsScene,
                             QGraphicsRectItem, QGraphicsPixmapItem, QFormLayout, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QMouseEvent
from PyQt5.QtCore import Qt, QRectF, QPointF, QRect

class ImageCropper(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Cropper and Box File Creator')
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        mainLayout = QVBoxLayout()

        # Upload Button
        self.uploadBtn = QPushButton('Upload Folder')
        self.uploadBtn.clicked.connect(self.uploadFolder)
        mainLayout.addWidget(self.uploadBtn)

        # Crop Button
        self.cropBtn = QPushButton('Crop Images')
        self.cropBtn.clicked.connect(self.cropImages)
        self.cropBtn.setEnabled(False)
        mainLayout.addWidget(self.cropBtn)

        # Create Box Files Button
        self.createBoxBtn = QPushButton('Create Box Files')
        self.createBoxBtn.clicked.connect(self.createBoxFiles)
        self.createBoxBtn.setEnabled(False)
        mainLayout.addWidget(self.createBoxBtn)

        # Setup form layout for cropping area and box file editing
        self.formLayout = QFormLayout()
        self.imageLabel = QLabel('No image loaded')
        self.formLayout.addRow(self.imageLabel)

        # Main container widget
        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

    def uploadFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.folderPath = folder
            self.imageFiles = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
            self.newFolder = os.path.join(folder, 'cropped_images')
            os.makedirs(self.newFolder, exist_ok=True)
            self.cropBtn.setEnabled(True)

    def cropImages(self):
        self.cropWindow = CropWindow(self.imageFiles, self.newFolder)
        self.cropWindow.show()

    def createBoxFiles(self):
        # Create box files logic here
        pass

class CropWindow(QWidget):
    def __init__(self, imageFiles, newFolder):
        super().__init__()

        self.imageFiles = imageFiles
        self.newFolder = newFolder
        self.currentIndex = 0
        self.startPos = QPointF()
        self.endPos = QPointF()
        self.rectItem = None  # Initialize rectItem
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Crop Images')
        self.setGeometry(150, 150, 800, 600)

        # Main layout
        mainLayout = QVBoxLayout()

        # Graphics View for Image
        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        mainLayout.addWidget(self.view)

        # Load first image
        self.loadImage()

        # Crop Button
        self.cropBtn = QPushButton('Crop and Save')
        self.cropBtn.clicked.connect(self.cropAndSave)
        mainLayout.addWidget(self.cropBtn)

        # Navigation Buttons
        self.prevBtn = QPushButton('Previous')
        self.prevBtn.clicked.connect(self.prevImage)
        mainLayout.addWidget(self.prevBtn)

        self.nextBtn = QPushButton('Next')
        self.nextBtn.clicked.connect(self.nextImage)
        mainLayout.addWidget(self.nextBtn)

        self.setLayout(mainLayout)

        # Set up mouse event handling
        self.view.setMouseTracking(True)
        self.view.mousePressEvent = self.startRect
        self.view.mouseMoveEvent = self.updateRect
        self.view.mouseReleaseEvent = self.endRect

    def loadImage(self):
        if 0 <= self.currentIndex < len(self.imageFiles):
            imagePath = self.imageFiles[self.currentIndex]
            pixmap = QPixmap(imagePath)
            self.scene.clear()
            self.imageItem = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.imageItem)
            self.view.setScene(self.scene)
        else:
            self.imageLabel.setText('No images to display')

    def startRect(self, event):
        if self.imageItem:
            self.startPos = self.view.mapToScene(event.pos())
            if self.rectItem:  # Remove any existing rectangle
                self.scene.removeItem(self.rectItem)
            self.rectItem = QGraphicsRectItem(QRectF())
            self.rectItem.setPen(Qt.red)
            self.scene.addItem(self.rectItem)

    def updateRect(self, event):
        if self.rectItem:
            self.endPos = self.view.mapToScene(event.pos())
            rect = QRectF(self.startPos, self.endPos).normalized()
            self.rectItem.setRect(rect)

    def endRect(self, event):
        if self.rectItem:
            self.endPos = self.view.mapToScene(event.pos())
            self.rectItem.setRect(QRectF(self.startPos, self.endPos).normalized())

    def cropAndSave(self):
        if self.rectItem:
            rect = self.rectItem.rect().toRect()
            if rect.isValid():
                pixmap = self.imageItem.pixmap().copy(rect)
                output_path = os.path.join(self.newFolder, os.path.basename(self.imageFiles[self.currentIndex]))
                pixmap.save(output_path)

                # Optionally move to the next image
                self.nextImage()

    def nextImage(self):
        self.currentIndex += 1
        if self.currentIndex < len(self.imageFiles):
            self.loadImage()
        else:
            self.close()

    def prevImage(self):
        self.currentIndex -= 1
        if self.currentIndex >= 0:
            self.loadImage()
        else:
            self.close()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageCropper()
    ex.show()
    sys.exit(app.exec_())
