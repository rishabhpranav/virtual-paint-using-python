# PyQt library 
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

# Mouse control library
import pyautogui
pyautogui.FAILSAFE = False
# CV logic 
import cv  # CV component
import cv2 # OpenCV


import math
import numpy as np
from scipy import spatial
from datetime import datetime
import os # for OS calls
import sys

# Setting up GUI attributes
WIDTH = 320*3
HEIGHT = 240*3
FONT = QtGui.QFont("HelveticaNeue",8)
rect = cv.rect # Used to map CV information to GUI
pipe = None # Used to send data from CV component to GUI

class Thread(QThread):
    """
    PyQt threads to allow the GUI to receive information from CV component using
    pipes
    """

    coordinateData = pyqtSignal()  # Data signal 

    def __init__(self):
        QThread.__init__(self)
        self.history = [] # Holds coordinate data for GUI to access

    def run(self):
        """
        Function running on thread to check for inputs from the pipe. Blocks
        until there is information from the pipe and sends a signal to the GUI. 
        Data is moved from the pipe to the history attribute for GUI to read.
        """
        while(True):
            if (pipe is not None):
                output = pipe.recv()
                self.history.append(output)
                self.coordinateData.emit()
                QtWidgets.QApplication.processEvents()


class MainWindow(QtWidgets.QMainWindow):
    """
    Creates the visual GUI and all assocciated button and features. GUI has two 
    'pages' where the initial page is the canvas section for users to draw on,
    and the second page is the toolbar section, containing all the tools users 
    can select. The CV component will send data to the GUI based on hand 
    gestures and act as the 'mouse' to manipulate the canvas.

    GUI allows users to do the following:
    - Draw 
    - Erase
    - Change pen sizes between three set pen sizes (must be done through hand
      gestures)
    - Create a new page on the canvas 
    - Save image 
    - Change pen colors (with the color and luminiosity picker)
    """

    def __init__(self):
        super().__init__()
        self.font = FONT
        self.setWindowTitle("Air Painter")
        self.tool_width = int(min(HEIGHT,WIDTH)*0.4)

        # GUI Attributes
        self.curr_state = 0 # Tracks state of the program
        self.all_tools = ["pen","eraser","newpage","save","color"]
        self.icon_size = WIDTH*0.06
        self.radius = int(min(HEIGHT,WIDTH)*0.4)//2
        self.luminosity_size = (self.icon_size//2,self.icon_size)
        self.padding = int(self.icon_size *0.25)

        # Toolbar related attributes
        self.drawing_mode = False
        self.pen_size = 3 # 8, 15
        self.pen_color = Qt.black
        self.draw_color = self.pen_color
        self.past_event = None # Last coordinate clicked
        self.curr_tool = "pen"

        # Initialize the canvas and tool bar sections        
        self.initialize()
        
        # Initializing thread to get "mouse" data
        self.th = Thread()
        self.th.coordinateData
        self.th.start()
        self.th.coordinateData.connect(self.fakeMousePressEvent,Qt.QueuedConnection)
        

    def setCommandLabels(self, text):
        """
        Simultaneously changes for the labels in the canvas and toolbar layouts
        to display what the last command executed was.
        """
        self.command_label.setText(text)
        self.commandT_label.setText(text)

    def pen_action_click(self, s):
        """
        Sets the pen to the last changed pen color. This pen color is displayed 
        in the toolbar.
        """
        self.curr_tool = "pen"
        self.draw_color = self.pen_color
        self.setCommandLabels("Using pen.")

    def erase_action_click(self, s):
        """
        Sets the pen color to white, which is the canvas background color.
        Erases in the canvas
        """
        self.curr_tool = "eraser"
        self.draw_color = QtGui.QColor(255, 255, 255, 255)
        self.setCommandLabels("Using eraser.")

    def newpage_click(self,s):
        """
        Deletes the canvas and resets it to a blank white screen.
        """
        canvas = QtGui.QPixmap(WIDTH, HEIGHT-self.icon_size)
        canvas.fill(Qt.white)
        self.canvas = canvas
        self.canvas_label.setPixmap(canvas)
        self.setCommandLabels("Generated new page.")

    def save_click(self,s):
        """
        Saves the canvas as a jpg image based on the current date and time
        """
        date = datetime.now()
        filename = date.strftime('%Y-%m-%d_%H-%M-%S.jpg')
        self.canvas_label.pixmap().save(filename)
        self.setCommandLabels("Saving image as "+filename)

    def menu_open(self,s):
        """
        Switches from the canvas layout to the toolbar menu layout. 
        """
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+1)
        self.setCommandLabels("Moving to toolbar menu")

    def menu_close(self,s):
        """
        Switches from the toolbar menu layout to the canvas layout.
        """
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()-1)
        self.setCommandLabels("Moving to canvas menu")

    def convert_pixel_to_rgb(self, x, y, wheel):
        """
        Takes in a canvas x,y coordinate and determines the color of the pixel
        at that location. 

        If wheel is true, then we want to determine the color 
        from a location in the color wheel and redraw the luminosity bar to 
        represent this color. We also update the pen color and associated pen 
        color display. 

        If wheel is false, then we are only updating the pen color and 
        associated pen color display. 
        """
        if (wheel == False): # Color selected from luminosity bar
            # Updating pen color and pen color display
            img = self.luminosity_label.pixmap().toImage()
            color = QtGui.QColor(img.pixel(x,y))
            icon = QtGui.QPixmap(self.icon_size, self.icon_size)
            icon.fill(color)
            self.color_icon.setPixmap(icon)
            self.draw_color = color
        else: # Color selected from color wheel
            # Updating pen color and pen color display
            img = self.color_wheel_label.pixmap().toImage()
            color = QtGui.QColor(img.pixel(x,y))
            icon = QtGui.QPixmap(self.icon_size, self.icon_size)
            icon.fill(color)
            self.color_icon.setPixmap(icon)
            self.draw_color = color

            # Updates the luminosity bar
            luminosity_bar = QtGui.QPixmap(self.tool_width, self.icon_size)
            luminosity_bar.fill(Qt.transparent)
            self.luminosity_label.setPixmap(luminosity_bar)
            self.createLuminosity()
        self.pen_color = color

        if (self.curr_tool == "eraser"):
            self.draw_color = Qt.white 
        # Display last executed command
        self.setCommandLabels("Changing pen color.")

    def mapCVtoGlobal(self, canvas_coord):
        """
        Assumes that canvas_coord is in bounds. In other words the x coordinate is in 
        the range [0, WIDTH] and the y coordinate is in the range [0, HEIGHT].
        """
        screen_loc = (self.geometry().x(), self.geometry().y())
        cv_width = rect[1][0]-rect[0][0]
        cv_height = rect[1][1]-rect[0][1]
        new_x =((canvas_coord[0]-rect[0][0])/(cv_width)*(WIDTH)) + screen_loc[0]
        new_y =((canvas_coord[1]-rect[0][1])/(cv_height)*(HEIGHT)) + screen_loc[1]

        return new_x, new_y

    def mapCVtoCanvas(self, canvas_coord):
        """
        Assumes that canvas_coord is in bounds. In other words the x coordinate is in
        the range [0, WIDTH] and the y coordinate is in the range [0, HEIGHT].
        """
        cv_width = rect[1][0] - rect[0][0]
        cv_height = rect[1][1] - rect[0][1]
        new_x = ((canvas_coord[0] - rect[0][0]) / (cv_width) * (WIDTH))
        new_y = ((canvas_coord[1] - rect[0][1]) / (cv_height) * (HEIGHT))
        return new_x, new_y

    def create_buttons(self):
        """
        Function creates the five buttons back, paint, erase, new page, and 
        save icons on the screen. 

        Back     - Goes from toolbar to canvas layout
        Paint    - Allows user to draw on the canvas
        Erase    - Allows user to erase on the canvas
        New Page - Resets the canvas to a blank white canvas
        Save     - Saves the canvas to an image 
        """
        back_action = QtWidgets.QAction("Back", self) 
        back_icon = QtGui.QIcon("images/back.svg")
        back_action.setIcon(back_icon)
        back_action.triggered.connect(self.menu_close)
        back_action.setFont(self.font)

        paint_action = QtWidgets.QAction("Paint", self) 
        paint_icon = QtGui.QIcon("images/paint.svg")
        paint_action.setIcon(paint_icon)
        paint_action.triggered.connect(self.pen_action_click)
        paint_action.setFont(self.font)
        
        erase_action = QtWidgets.QAction("Eraser", self) 
        erase_icon = QtGui.QIcon("images/eraser.svg")
        erase_action.setIcon(erase_icon)
        erase_action.triggered.connect(self.erase_action_click)
        erase_action.setFont(self.font)

        newpage_action = QtWidgets.QAction("New Page", self) 
        newpage_icon = QtGui.QIcon("images/new-page.svg")
        newpage_action.setIcon(newpage_icon)
        newpage_action.triggered.connect(self.newpage_click)
        newpage_action.setFont(self.font)

        save_action = QtWidgets.QAction("Save", self) 
        save_icon = QtGui.QIcon("images/save.svg")
        save_action.setIcon(save_icon)
        save_action.triggered.connect(self.save_click)
        save_action.setFont(self.font)

        # Saves the tools to a list
        self.tools_list = [back_action, paint_action, erase_action, 
                           newpage_action, save_action]

    def setLayoutStyling(self, layout):
        """
        Used to ensure that there is no padding and remove margins when
        setting up layouts
        """
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def initialize(self):
        """
        Initializes GUI structure, which consists of a canvas and toolbar 
        layout. The canvas layout shows up by default and by selecting buttons
        you can switch between the canvas and toolbar layout. 
        """
        ### Main container for all the sections ###
        self.main_container = QtWidgets.QFrame(self)
        self.main_container.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.main_layout = QtWidgets.QHBoxLayout(self.main_container)
        self.setCentralWidget(self.main_container)
        self.stackedWidget = QtWidgets.QStackedWidget()
        
        # Main comtainer styling
        self.setLayoutStyling(self.main_layout)


        ### Canvas Section ###
        # Containers for canvas section
        canvas_padding_container = QtWidgets.QFrame(self)
        padding_canvas_layout = QtWidgets.QVBoxLayout(canvas_padding_container)

        canvas_container = QtWidgets.QFrame(self)
        canvas_layout = QtWidgets.QVBoxLayout(canvas_container)
        self.canvas_container = canvas_container

        canvas_subcontainer = QtWidgets.QFrame(self)
        canvas_sublayout = QtWidgets.QHBoxLayout(canvas_subcontainer)

        self.setLayoutStyling(padding_canvas_layout)
        self.setLayoutStyling(canvas_layout)
        self.setLayoutStyling(canvas_sublayout)

        # Actual Canvas
        self.canvas_label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(WIDTH, HEIGHT-self.icon_size)
        canvas.fill(Qt.white)
        self.canvas_label.setPixmap(canvas)
        self.canvas = canvas

        # Command Bar
        self.command_label = QtWidgets.QLabel("Starting canvas...")
        self.command_label.setStyleSheet("color:white; background-color:black;")
        self.command_label.setFont(self.font)
        self.command_label.setFixedWidth(WIDTH-self.icon_size)
        self.command_label.setFixedHeight(self.icon_size)

        # Menu button
        menu_action = QtWidgets.QAction("Menu", self) 
        menu_icon = QtGui.QIcon("images/tools.svg")
        menu_action.setIcon(menu_icon)
        menu_action.triggered.connect(self.menu_open)

        canvas_menubar = self.addToolBar("")
        canvas_menubar.setFixedSize((QtCore.QSize(self.icon_size,self.icon_size)))
        canvas_menubar.addAction(menu_action)
        canvas_menubar.setIconSize(QtCore.QSize(self.icon_size,self.icon_size))
        canvas_menubar.setStyleSheet("padding:0px;")


        # Adding widgets to canvas layout section
        padding_canvas_layout.addWidget(self.canvas_label)
        canvas_layout.addWidget(canvas_padding_container)
        canvas_sublayout.addWidget(self.command_label)
        canvas_sublayout.addWidget(canvas_menubar)
        canvas_layout.addWidget(canvas_subcontainer)

        #self.setLayoutStyling(canvas_menubar)
        canvas_menubar.setContentsMargins(0, 0, 0, 0)
        self.command_label.setContentsMargins(0, 0, 0, 0)

        ### Toolbar Section ###
        # Containers for toolbar section
        tool_container = QtWidgets.QFrame(self)
        tool_layout = QtWidgets.QVBoxLayout(tool_container)

        padding_tool_container = QtWidgets.QFrame(self)
        padding_tool_layout = QtWidgets.QVBoxLayout(padding_tool_container)
        self.padding_tool_container = padding_tool_container

        curr_color_container = QtWidgets.QFrame(self)
        curr_color_layout = QtWidgets.QHBoxLayout(curr_color_container)

        self.setLayoutStyling(tool_layout)
        self.setLayoutStyling(padding_tool_layout)
        self.setLayoutStyling(curr_color_layout)

        # Creates all the buttons
        self.create_buttons() 
        menubar = self.addToolBar("")
        menubar.setFixedHeight(self.icon_size*2)
        menubar.setIconSize(QtCore.QSize(self.icon_size,self.icon_size))
        menubar.setStyleSheet("padding:0px;")
        menubar.setContentsMargins(0, self.icon_size//2, 0, 0)

        right_spacer = QtWidgets.QWidget()
        right_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Expanding)
        left_spacer = QtWidgets.QWidget()
        left_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Expanding)
        
        menubar.addWidget(right_spacer)
        for x in self.tools_list:
            menubar.addAction(x)
        menubar.addWidget(left_spacer)
        self.menubarHeight = menubar.height()

        # Color picker 
        self.color_wheel_label = QtWidgets.QLabel()
        color_wheel = QtGui.QPixmap(self.tool_width, self.tool_width)
        color_wheel.fill(Qt.transparent)
        self.color_wheel_label.setPixmap(color_wheel)
        color_picker = self.create_color_wheel()
        self.color_wheel_label.setAlignment(QtCore.Qt.AlignCenter)
        self.color_wheel_label.setFixedHeight(self.radius*2+self.padding*2)

        # Luminosity bar
        self.luminosity_label = QtWidgets.QLabel()
        luminosity_bar = QtGui.QPixmap(self.tool_width, self.icon_size)
        luminosity_bar.fill(Qt.transparent)
        self.luminosity_label.setPixmap(luminosity_bar)
        self.luminosity_label.setFixedHeight(self.icon_size+self.padding*2)
        self.createLuminosity()
        self.luminosity_label.setAlignment(QtCore.Qt.AlignCenter)
        self.luminosity_label.setStyleSheet("padding:0px;")

        # Label to show the current pen color
        curr_color_label = QtWidgets.QLabel("Pen Color")
        curr_color_label.setFont(self.font)
        curr_color_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        curr_color_label.setStyleSheet("padding-right:5px;")
        curr_icon_label = QtWidgets.QLabel()
        curr_icon_label.setStyleSheet("spacing:0px") # todo font not changing
        curr_color_icon = QtGui.QPixmap(self.icon_size, self.icon_size)
        curr_color_icon.fill(Qt.black)
        curr_icon_label.setPixmap(curr_color_icon)
        self.color_icon = curr_icon_label

        # Displays last executed command
        self.commandT_label = QtWidgets.QLabel("Starting canvas...")
        self.commandT_label.setStyleSheet("background-color:black;color:white;")
        self.commandT_label.setFixedHeight(self.command_label.height())
        self.commandT_label.setFixedWidth(WIDTH)
        self.commandT_label.setFont(self.font)


        # Adding widgets to toolbar layout section
        padding_tool_layout.addWidget(self.color_wheel_label,Qt.AlignTop)
        padding_tool_layout.addWidget(self.luminosity_label, Qt.AlignTop)

        curr_color_layout.addWidget(curr_color_label, Qt.AlignVCenter)
        curr_color_layout.addWidget(curr_icon_label, Qt.AlignVCenter)

        padding_tool_layout.addWidget(curr_color_container, Qt.AlignTop)
        padding_tool_layout.addWidget(menubar)

        padding_tool_layout.addWidget(self.commandT_label)

        tool_layout.addWidget(padding_tool_container)

        # Combining canvas and toolbar layouts
        self.stackedWidget.addWidget(canvas_container)
        self.stackedWidget.addWidget(tool_container)
        self.main_layout.addWidget(self.stackedWidget)

        # Offsetting wheel and luminosity pixel selection for mouse click
        self.wheel_offset = (-WIDTH//2 + self.radius, 
            -(self.radius + self.padding) + self.radius)

        self.lum_offset = (-WIDTH//2 + self.tool_width//2, 
            -(self.radius*2+ self.padding*3 ))
     
    def create_color_wheel(self):
        """
        Draws a color wheel on the toolbar screen. The component will be used 
        as a color picker.
        https://stackoverflow.com/questions/50720719/how-to-create-a-color-circle-in-pyqt 
        was referenced to create the color wheel.
        """
        radius = self.radius
        painter = QtGui.QPainter(self.color_wheel_label.pixmap())
        pen = QtGui.QPen()

        for i in range(radius*2): #width
            for j in range(radius*2): #height
                color = QtGui.QColor(255, 255, 255, 255)
                h = (np.arctan2(i-radius, j-radius)+np.pi)/(2.*np.pi)
                s = np.sqrt(np.power(i-radius, 2)+np.power(j-radius, 2))/radius
                v = 1.0
                if s < 1.0:
                    color.setHsvF(h, s, v, 1.0)
                    pen.setColor(color)
                    painter.setPen(pen)
                    painter.drawPoint(i,j)

        painter.end()

    def createLuminosity(self):
        """
        Creates the luminosity bar. Shows a few variation of color by varying
        the luminosity. Only shows a certain amount of colors in the luminosity
        page depending on the allowed width for display.
        """
        # Initializing painter and pen
        painter = QtGui.QPainter(self.luminosity_label.pixmap())
        pen = QtGui.QPen()

        length = self.tool_width
        hsl = QtGui.QColor(self.draw_color).getHsl()
        hue, sat = hsl[0], hsl[1]
        w, h = self.luminosity_size

        # Calculations to draw the luminosity bar
        bar_number = int(length//w)
        increment = int(math.ceil(255./(bar_number-1)))
        lum = 255
        color = QtGui.QColor(255, 255, 255, 255)
        x, y = 0,0
        for i in range(bar_number):
            color.setHsl(hue, sat, max(lum-i*increment, 0))
            pen.setColor(color)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(color, Qt.SolidPattern))
            painter.drawRect(x+i*(w), y, w,h)
        painter.end()

    def checkInBounds(self, x, y):
        """
        Check if position x,y is inside the color wheel
        """
        distance = np.sqrt((x-self.radius)**2 + (y-self.radius)**2)

        if (distance <self.radius): return True
        return False

    def mousePressEvent(self, QMouseEvent):
        """
        Mouse is clicked or pressed.
        """
        # Starts drawing mode
        self.drawing_mode = True
        self.past_event  = (QMouseEvent.pos().x(), QMouseEvent.pos().y())
        padding_coordinate = self.padding_tool_container.mapFromParent(QMouseEvent.pos())
        coord = QMouseEvent.pos()

        # Coordinates will differ depending on settings
        offset_coord1 = (coord.x()+self.wheel_offset[0],
                         coord.y()+self.wheel_offset[1])
        offset_coord2 = (coord.x()+self.lum_offset[0], 
                         coord.y()+self.lum_offset[1])
        
        # Detects clicks on color wheel or luminosity bar when in tool layout
        if self.stackedWidget.currentIndex() == 1: # only check in menu state
            if self.checkInBounds(offset_coord1[0], offset_coord1[1]):
                self.convert_pixel_to_rgb(offset_coord1[0], offset_coord1[1],True)

            elif(offset_coord2[0]>0 and offset_coord2[0]<self.tool_width  and 
                offset_coord2[1]>0) and offset_coord2[1]<self.icon_size:
                self.convert_pixel_to_rgb(offset_coord2[0], offset_coord2[1],False)
    
    @pyqtSlot()
    def fakeMousePressEvent(self):
        """
        Function is called when data is inputted from the pipe. Function simulates a 
        mouse press based on coordinate information from self.th.history. The attribute
        also contains information on what state the program should be in.
        """
        # Starts drawing mode
        history = self.th.history
        while len(history) > 0:
            # Gets coordinat information and maps data to canvas screen
            cv_coord, state = history[0]
            coord = self.mapCVtoGlobal(cv_coord)
            history.pop(0)
            # State does not make any changes that need to be displayed
            if (state != 11 and self.curr_state == 11):
                self.mouseReleaseEvent(coord)
                
            if (state == self.curr_state and state != 11 and state!=1):
                continue

            # Do nothing state. Ensure that mouse is released
            if (state == 0):
                self.curr_state = state

            # Move mouse cursor and change state for first time entering cursor mode
            elif (state==1 and self.curr_state!=state):
                self.curr_state = state
                self.past_locations = []
                self.past_locations.append(coord)
                pyautogui.moveTo(coord)

            # In cursor mode and detects if the finger hovers over a button for a few seconds
            elif (state==1):
                self.past_locations.append(coord)
                self.past_locations = self.past_locations[-5:] # gotta tune
                if (len(self.past_locations)== 5):
                    distance = np.array(self.past_locations)
                    largest_d = np.max(spatial.distance.cdist(distance,distance))
                    if (largest_d<=20):
                        pyautogui.click(coord)
                        self.past_locations = []
                pyautogui.moveTo(coord)

            # Using smallest pen size
            elif (state==2):
                self.pen_size = 3
                self.setCommandLabels("Using smallest pen size.")
                self.curr_state = state

            # Using medium pen size
            elif (state==3):
                self.pen_size = 8
                self.setCommandLabels("Using medium pen size.")
                self.curr_state = state

            # Using largest pen size
            elif (state==4):
                self.pen_size = 15
                self.setCommandLabels("Using largest pen size.")
                self.curr_state = state

            # Toggles between pen and eraser
            elif (state==5 and self.curr_state!=state):
                if (self.curr_tool == "pen"):
                    self.erase_action_click(None)
                else:
                    self.pen_action_click(None)

                self.curr_state = state

            # Using pen and using mouse coordinates to move cursor
            elif (state==11 and self.curr_state!=state):
                self.curr_state = state
                pyautogui.moveTo(coord)
                pyautogui.click()

            # Draw or erase based on mouse coordinates
            elif (state==11):
                canvas_coord = self.mapCVtoCanvas(cv_coord)
                self.draw_something(canvas_coord, self.pen_size)
                pyautogui.moveTo(coord)

            # Processes events immediately on GUI to prevent update delays
            QtWidgets.QApplication.instance().processEvents()

    def mouseMoveEvent(self, QMouseEvent):
        """
        Mouse moves after it is being held down
        """
        coord = QMouseEvent.pos()
        coord = (coord.x(), coord.y())
        self.draw_something(coord, self.pen_size)

    def mouseReleaseEvent(self, QMouseEvent):
        """
        Mouse released after being held down on click
        """
        self.drawing_mode = False
        self.past_event = None

    def draw_something(self, curr_event, size):
        """
        Draws on the canvas by connecting the current point to the last 
        detected point. Takes in curr_event, which is the coordinate point and 
        size which determines the pen size.
        """
        painter = QtGui.QPainter(self.canvas_label.pixmap())
        if (self.past_event is None):
            self.past_event = curr_event
        past = (self.past_event[0], self.past_event[1])
        curr = (curr_event[0],curr_event[1])

        # Sets up canvas
        pen = QtGui.QPen()
        pen.setWidth(size)
        color =  self.draw_color
        pen.setColor(color)
        painter.setPen(pen)

        # Create line
        painter.drawLine(past[0],past[1],curr[0],curr[1])
        self.update()
        self.past_event = curr_event

        # Terminate painting
        painter.end()


def main(pipe_object):
    """
    Initializes GUI window 
    """
    global window, app, pipe
    pipe = pipe_object
    app = QtWidgets.QApplication(sys.argv)

    # Setting font
    _id = QtGui.QFontDatabase.addApplicationFont("helveticaneue/HelveticaNeue Light.ttf")

    # Running GUI
    window = MainWindow()
    window.show()
    app.exec_()