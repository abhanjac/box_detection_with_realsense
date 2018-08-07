#!/usr/bin/env python

import cv2, numpy as np, time, pickle, os
import pyrealsense as pyrs
import matplotlib.pyplot as plt
from skimage import feature, exposure
from sklearn.svm import SVC
import threading as th

#===============================================================================

class videoRecorder( object ):
    '''
    Records the video from the video capture object given as input.
    '''
    def __init__( self, videoCaptureObject=None, fps=0 ):
        # If no fps is specified then set the fps from the input object.
        self.fps = videoCaptureObject.get( cv2.CAP_PROP_FPS ) if fps==0 else fps      
        self.fourcc = cv2.VideoWriter_fourcc( 'M','P','E','G' )
        self.recordFrame = 0
        # This flag when 1, indicates that the recording command is given.
        
#-------------------------------------------------------------------------------

    def record( self, name, frame, recordFrame, overwrite=0 ):   
        # recordFrame is the command to start or stop recording.

        if recordFrame == ord('r') and self.recordFrame == 0: 
            # Initialize writer and start recording when r is pressed.
            
            self.recordFrame = 1    
            # Since this flag is 1 from now onwards, so even if the r is 
            # pressed again, this if will not be executed and hen the writer 
            # will not be reinintialized.
            
            nameOfFile = name + time.strftime('_%d%b%Y_%H_%M_%S') + '.avi' if \
                overwrite == 0 else name + '.avi' 
            # If overwrite is 0, video files will be created with time stamp so 
            # that they do not overwrite any previously saved files.
            
            row, col = frame.shape[0], frame.shape[1]
            self.writer = cv2.VideoWriter( nameOfFile, self.fourcc, self.fps, \
                (col, row) )
            #print( self.writer.isOpened() )
            self.writer.write( frame )  # Writing the frame.
            print( '\nRecording Started ...' )

        elif recordFrame == ord('s') and self.recordFrame == 1: 
            # Stop recording on pressing s (if writer was initialized before).
            self.recordFrame = 0
            self.writer.release()
            print( '\nStopped Recording ...' )

        elif self.recordFrame == 1:   
            # recordframe flag is 1 means that writer is already initialized.
            self.writer.write( frame )  # Writing the frame.        

        else:
            pass

#===============================================================================

class mvAvg( object ):
    '''
    This function calculates the moving average in a given window size. 
    It recalculates the average value with the receipt of every new element.
    '''
    def __init__(self, window=1):
        self.window, self.newElemIdx = window, 0
        self.arrayOfElements = np.zeros( self.window )

#-------------------------------------------------------------------------------

    def calc(self, newElement):
        # Storing the new element.
        self.arrayOfElements[ self.newElemIdx ] = newElement    
        # Updating the index to point to a new location where the next new 
        # element will be stored.
        self.newElemIdx = int( ( self.newElemIdx + 1 ) % self.window )  
        # Calculating the mean.
        self.mvAvgValue = np.mean( self.arrayOfElements )   
        return self.mvAvgValue
    
#===============================================================================

class BoxDetector( object ):
    def __init__( self, useRealsense=False, recordRawVideoFramsToo=True, \
                        useRos=False, filePath='./' ):
        '''
        Initializing the BoxDetector class.
        '''
        
#-------------------------------------------------------------------------------

        # Input variables.
        self.useRealsense = useRealsense
        self.recordRawVideoFramsToo = recordRawVideoFramsToo
        self.useRos = useRos
        self.filePath = filePath
        
#-------------------------------------------------------------------------------

        # imageDetection variables.
        self.ix, self.iy, self.fx, self.fy = -1, -1, -1, -1
        self.drawing, self.cursorX, self.cursorY = False, -1, -1
        
#-------------------------------------------------------------------------------

        # Record training data flag. If there is a folder called './frames' in 
        # current directory, then on pressing 't' this flag will be True and 
        # training data will be recorded.
        self.recordTrainingData = False
        self.key = ord('`')
        self.loopCondition = True    # To start the while loop.
        self.model = None       # The svm model object.
        self.camera = None      # The camera object.
        self.serv = None
        self.inputColorFrame = None
        self.height, self.width, self.channel = None, None, None
        
#-------------------------------------------------------------------------------

        # readFrames variables.
        self.originalColorFrame = None
        self.originalDepthFrame = None
        self.inputDepthFrame = None
        self.depthFrame2000mm = None
        self.depthFrame200cm = None
        #self.depthFrame = None
        
#-------------------------------------------------------------------------------

        # removeEndEffectorClutter variables.
        self.modifiedInputFrame = None
        self.UpUeS1, self.LpLeS1, self.SpS1, self.EpS1 = 190, 300, 0, 30
        self.MeS1 = int( (self.UpUeS1+self.LpLeS1) / 2 )         # Middle Edge.
        self.UpUeS2, self.LpLeS2, self.SpS2, self.EpS2 = 197, 270, 30, 65       
        self.MeS2 = int( (self.UpUeS2+self.LpLeS2) / 2 )         # Middle Edge.
        self.UpUeS3, self.LpLeS3, self.SpS3, self.EpS3 = 201, 270, 65, 95       
        self.MeS3 = int( (self.UpUeS3+self.LpLeS3) / 2 )         # Middle Edge.
        self.UpUeS4, self.LpLeS4, self.SpS4, self.EpS4 = 208, 260, 95, 207       
        self.MeS4 = int( (self.UpUeS4+self.LpLeS4) / 2 )         # Middle Edge.
        
#-------------------------------------------------------------------------------

        # preProcess variables.
        self.combinedMask = None
        self.contours3 = []
        self.outputLabel = 0
        self.x, self.y, self.w, self.h = -1, -1, -1, -1
        
        # Parameters for the pre-processing stage.
        self.edge, self.kernelSize, self.structElemSize = 18, 3, 5
        self.minSize, self.maxSize = 210, 5000
        self.minArea, self.maxArea = 2000, 75000
        self.lwAspectRatioLimit, self.upAspectRatioLimit = 1.0, 1.3

        # The following lower and upper thresholds of color are in BGR format.
        self.lwColorThresh = np.array([60,70,60])
        self.upColorThresh = np.array([140,150,130])
        
        # Parameters for the histogram.
        self.pixelsPerCell, self.cellsPerBlock = (12,12), (5,5)

#-------------------------------------------------------------------------------

        # findDistance variables.
        self.X, self.Y, self.Z = 0.0, 0.0, 0.0
        self.Xstr, self.Ystr, self.Zstr = 'N/A', 'N/A', 'N/A'
        
        # Initializing the flag that indicates what should be used as the 
        # source of depth measurement.
        self.useRealsenseForDepth = False

        # The depth is not very reliable (from bgr image or depth image) 
        # if the vehicle is closer than this distance (in mm) to the object.
        self.minDistToDisplayDepth = 750     

        # Min, max ranges (in mm) within which realsense detects proper depth.
        self.realsenseMinReliableRange = 500
        self.realsenseMaxReliableRange = 1500    

        # Approximate focal length of realsense. This is calculated by putting
        # an object of a known dimension in front of the camera.

        # Distance (actual (mm)), Width (actual (mm)), Pixel width.
        self.D, self.W, self.P = 610, 36, 34
        
        # Rough focal length of realsense (pixels).
        self.F = self.P * self.D * 1.0 / self.W
        
        # Actual physical height and width of the door (mm).
        self.doorWidth, self.doorHeight = 262, 314    
        
        self.numOfBins = 250     # Number of bins of histogram of depth image.
        self.maxHistValueRange = 2500   # Max value of pixel range in histogram.
        
#-------------------------------------------------------------------------------

        # drawCoordinates variables.
        self.state = ''
        
#-------------------------------------------------------------------------------

        # Load svm model.
        modelFile = os.path.join( self.filePath, 'model.p' )
        if os.path.exists( modelFile ):
            f = open( modelFile, 'rb' )
            self.model = pickle.load( f )
            f.close()
        else:
            print( 'No svm model found. Aborting.' )
            exit(1)

#-------------------------------------------------------------------------------

        # Initializing the cameras.
        if self.useRealsense == True:
            # Starting the realsense.
            # start service (also available as context manager)
            self.serv = pyrs.Service()   
            Streams = [ pyrs.stream.ColorStream( fps = 60, color_format = \
                                                'bgr' ), \
                        pyrs.stream.DepthStream( fps = 60 ), \
                        pyrs.stream.DACStream( fps = 60 ) ]    
            # create a device from device id and streams of interest.
            
            # The DACStream is a depth aligned to color stream where the depth 
            # image pixels are exactly aligned to the color pixels.
            self.camera = self.serv.Device( device_id = 0, streams = Streams )

        else:
            #self.camera = cv2.VideoCapture(0)
            self.camera = cv2.VideoCapture( \
            './raw_21Mar2018_11_26_32.avi' )
            
#===============================================================================

    def readFrames( self ):
        '''
        Reads the input frames.
        '''
        if self.useRealsense:
            realsense = self.camera
            realsense.wait_for_frames()
            self.originalColorFrame = realsense.color
            
            # This frame has the depth values in mm. This is actually the dac 
            # frame where the depth map is aligned to the color frames.
            self.originalDepthFrame = realsense.dac       
            
            # We are never touching the originalDepthFrame, 
            # always working with a copy of that one.
            self.inputDepthFrame = np.array( self.originalDepthFrame )      
            
            # Converting the depthFrame into cm resolution so that it can be 
            # represented as uint8 (required for displaying properly, 
            # colormap and recording as video).
            
            # Clipping all values beyond 2m as the realsense is only best at 
            # seeing between 500mm to 1500mm (or max to 2500mm).
            self.depthFrame2000mm = np.clip( self.inputDepthFrame, 0, 2500 )    
            
            # Converting pixel values to cm so that 2500mm can become 250cm and 
            # thereby represented as 8 bit int.
            self.depthFrame200cm = np.asarray( self.depthFrame2000mm / 10.0, \
                                          dtype=np.uint8 )   
            
            ## Preparing this frame for recording depth video, 
            ## so it has to be of 3 channel.
            #self.depthFrame = cv2.cvtColor( self.depthFrame200cm, cv2.COLOR_GRAY2BGR )     
            #self.depthFrame = cv2.applyColorMap( self.depthFrame, cv2.COLORMAP_JET )
            
        else:
            cam = self.camera
            ret, self.originalColorFrame = cam.read()
            if ret == False:
                print( 'No frame found from camera. Aborting.' )
                exit(1)

#===============================================================================

    def removeEndEffectorClutter( self ):
        '''
        Removes the clutter due to the end effector from the frame.
        '''
        if self.inputColorFrame is None:
            return
        
        self.modifiedInputFrame = np.array( self.inputColorFrame )
        # Upper Part Upper Edge, Lower Part Lower Edge, Start Pixel, End Pixel 
        # (of Section 1 from left boundary of frame).
        self.modifiedInputFrame[ self.UpUeS1 : self.MeS1, self.SpS1 : self.EpS1, : ] = \
            np.vstack( [ self.modifiedInputFrame[ self.UpUeS1 : self.UpUeS1 + 1, \
                self.SpS1 : self.EpS1, : ] ] * ( self.MeS1 - self.UpUeS1 ) )    
        # Removing the leftmost part.
        
        self.modifiedInputFrame[ self.MeS1 : self.LpLeS1, self.SpS1 : self.EpS1, : ] = \
            np.vstack( [ self.modifiedInputFrame[ self.LpLeS1 : self.LpLeS1 + 1, \
                self.SpS1 : self.EpS1, : ] ] * ( self.LpLeS1 - self.MeS1 ) )
        
        # Upper Part Upper Edge, Lower Part Lower Edge, Start Pixel, End Pixel 
        # (of Section 2 from left boundary of frame).
        self.modifiedInputFrame[ self.UpUeS2 : self.MeS2, self.SpS2 : self.EpS2, : ] = \
            np.vstack( [ self.modifiedInputFrame[ self.UpUeS2 : self.UpUeS2 + 1, \
                self.SpS2 : self.EpS2, : ] ] * ( self.MeS2 - self.UpUeS2 ) )    
        # Removing the second leftmost part.
        
        self.modifiedInputFrame[ self.MeS2 : self.LpLeS2, self.SpS2 : self.EpS2, : ] = \
            np.vstack( [ self.modifiedInputFrame[ self.LpLeS2 : self.LpLeS2 + 1, \
                self.SpS2 : self.EpS2, : ] ] * ( self.LpLeS2 - self.MeS2 ) )
        
        # Upper Part Upper Edge, Lower Part Lower Edge, Start Pixel, End Pixel 
        # (of Section 3 from left boundary of frame).
        self.modifiedInputFrame[ self.UpUeS3 : self.MeS3, self.SpS3 : self.EpS3, : ] = \
            np.vstack( [ self.modifiedInputFrame[ self.UpUeS3 : self.UpUeS3 + 1, \
                self.SpS3 : self.EpS3, : ] ] * ( self.MeS3 - self.UpUeS3 ) )    
        # Removing the third leftmost part.
        
        self.modifiedInputFrame[ self.MeS3 : self.LpLeS3, self.SpS3 : self.EpS3, : ] = \
            np.vstack( [ self.modifiedInputFrame[ self.LpLeS3 : self.LpLeS3 + 1, \
                self.SpS3 : self.EpS3, : ] ] * ( self.LpLeS3 - self.MeS3 ) )
        
        # Upper Part Upper Edge, Lower Part Lower Edge, Start Pixel, End Pixel
        # (of Section 4 from left boundary of frame).
        self.modifiedInputFrame[ self.UpUeS4 : self.MeS4, self.SpS4 : self.EpS4, : ] = \
            np.vstack( [ self.modifiedInputFrame[ self.UpUeS4 : self.UpUeS4 + 1, \
                self.SpS4 : self.EpS4, : ] ] * ( self.MeS4 - self.UpUeS4 ) )    
        # Removing the fourth leftmost part.
        
        self.modifiedInputFrame[ self.MeS4 : self.LpLeS4, self.SpS4 : self.EpS4, : ] = \
            np.vstack( [ self.modifiedInputFrame[ self.LpLeS4 : self.LpLeS4 + 1, \
                self.SpS4 : self.EpS4, : ] ] * ( self.LpLeS4 - self.MeS4 ) )
        
#===============================================================================

    def preProcess( self ):
        '''
        Pre-pocessing the frames to get a good contour detection.
        Creating a mask based on contours.
        Creating a mask based on color.
        Second contour detection stage.
        Filtering the bounding rectangle by aspect ratio and applying svm.
        '''
        if self.inputColorFrame is None:
            return
        
#-------------------------------------------------------------------------------
        
        # Pre-pocessing the frames to get a good contour detection.

        startingFrame = np.array( self.modifiedInputFrame )
        grayFrame = cv2.cvtColor( startingFrame, cv2.COLOR_BGR2GRAY )
        edgeFrame = cv2.Canny( grayFrame, threshold1=self.edge, threshold2=self.edge*3, \
            apertureSize=3 )
        
        blurFrame = cv2.blur( edgeFrame, ksize=(self.kernelSize, self.kernelSize) )
        binaryFrame = cv2.adaptiveThreshold( blurFrame, 255, \
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.kernelSize, 2 )
        
        hori = cv2.getStructuringElement( cv2.MORPH_RECT, \
            ksize=( self.structElemSize, self.structElemSize ) )
        
        # Making the edges thicker.
        dilateFrame = cv2.dilate( binaryFrame, kernel=hori, anchor=(-1,-1) )    
        
        # Finding the contours.
        _, contours, hierarchy = cv2.findContours( dilateFrame, cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_SIMPLE )
        
        # Filtering contours based on size.
        contours1 = [ c for c in contours if cv2.arcLength(c, False) < self.maxSize \
            and cv2.arcLength(c, False) > self.minSize \
            and cv2.contourArea(c) < self.maxArea and cv2.contourArea(c) > self.minArea ]   
        
#-------------------------------------------------------------------------------
        
        # Creating a mask based on contours.

        contourFrame = np.zeros( (self.height, self.width, self.channel), \
                                  dtype=np.uint8 )
        #cv2.imshow( 'contourFrame', contourFrame )
        
        # We need this mask to be of single channel.
        contourMask = np.zeros( (self.height, self.width), dtype=np.uint8 )   
        if len(contours1) != 0:
            # Drawing contours after filtering. 
            contourFrame1 = np.zeros( (self.height, self.width, self.channel), \
                                  dtype=np.uint8 )
            cv2.drawContours( contourFrame1, contours1, -1, (0,255,0), 3 )    
            #cv2.imshow( 'contourFrame1', contourFrame1 )

            # This will be used as a mask. Hence rectangles are drawn as (1).
            cv2.drawContours( contourMask, contours1, -1, (1), cv2.FILLED )    
            #cv2.imshow( 'contourMask', contourMask )

#-------------------------------------------------------------------------------

        # Creating a mask based on color.

        # Filtering by color. But the white regions have values 255 which needs
        # to be converted to 1.
        colorMask = cv2.inRange( startingFrame, self.lwColorThresh, \
                                                self.upColorThresh ) / 255.0
        #cv2.imshow( 'colorMask', colorMask )
        
        colorMask = np.asarray( colorMask, dtype=np.uint8 )
        self.combinedMask = contourMask * colorMask
        #combinedMask = np.asarray( self.combinedMask, dtype=np.uint8 )
        #cv2.imshow( 'Mask', self.combinedMask*255 )
        
        maskedFrame = grayFrame * self.combinedMask    # Masking input frame.
        
#-------------------------------------------------------------------------------

        # Second contour detection stage.

        # Finding the contours in the masked frame.
        _, contours2, _ = cv2.findContours( maskedFrame, cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_SIMPLE )    
        finalFrame = np.zeros( (self.height, self.width, self.channel), \
                                dtype=np.uint8 )
        
        # Filtering contours based on size.
        self.contours3 = [ c for c in contours2 if \
                          cv2.arcLength(c, False) < self.maxSize \
                      and cv2.arcLength(c, False) > self.minSize \
                      and cv2.contourArea(c) < self.maxArea \
                      and cv2.contourArea(c) > self.minArea ]    
        
        # Finding the dimensions of bounding rectangle around largest contour.
        (self.x, self.y, self.w, self.h) = cv2.boundingRect( \
                                            self.contours3[0] ) \
                if len(self.contours3) != 0 else (-1, -1, -1, -1)    
        
#-------------------------------------------------------------------------------

        # Filtering the bounding rectangle by aspect ratio and applying svm.

        # then finding the hog features and applying svm.
        if self.h*1.0/self.w > self.lwAspectRatioLimit and \
           self.h*1.0/self.w < self.upAspectRatioLimit:        
            svmInput = self.modifiedInputFrame[ self.y : self.y+self.h, 
                                                self.x : self.x+self.w, : ]
            
            # Shrinking the input image.
            svmInputSmall = cv2.resize( svmInput, (60, 72), cv2.INTER_AREA )    
            svmGrayInput = cv2.cvtColor( svmInputSmall, cv2.COLOR_BGR2GRAY ) 
            #cv2.imshow( 'croppedImg', svmGrayInput )          
            
            # Extracting the hog features.
            (hog, hogImg) = feature.hog( svmGrayInput, orientations=9, \
                pixels_per_cell=self.pixelsPerCell, cells_per_block=self.cellsPerBlock, \
                block_norm='L2-Hys', transform_sqrt=True, visualise=True )   
            #hogImg = exposure.rescale_intensity( hogImg, out_range=(0,255) )            
            #hogImg = hogImg.astype('uint8')
            #cv2.imshow('image', hogImg)

            # Converting (450,) shape into (1, 450) shape to feed it into model.
            hog = np.array( [hog] )     
            
            # Predicting label. Correct object detection will make this flag 1.
            self.outputLabel = self.model.predict( hog )      
            #outputLabel = 1   # Making this outputLabel = 1 will shut down svm.
        else:
            # If the bounding rectangle is not of proper aspect ratio, then 
            # reset the corner pixel and height and width values to -1 as 
            # set by default.
            self.x, self.y, self.w, self.h = -1, -1, -1, -1

#===============================================================================        

    def findDistance( self ):
        '''
        Drawing rectangles and measuring distances when svm predicts positive.
        Fixing the angle of tilt of rectangle and calculating the depth.
        Calculate the distance from the size of the box.
        Calculate the distance from realsense.
        When inside the range of 500 to 1500mm from the object, we 
        switch to realsense to calculate the depth X.
        Within 750mm from the door, no more rectangle display is needed 
        as the approach will be head-on.
        '''
        if self.inputColorFrame is None:
            return
        
        # Initializing the rectangle tilt angle.
        rectAngle = 0.0
            
#-------------------------------------------------------------------------------

        # Drawing rectangles and measuring distances when svm predicts positive.
        
        if self.outputLabel == 1:    # svm predicts that the object is detected.
            #print(x,y,w,h,outputLabel)
            
            self.outputLabel = 0   # Reset outputLabel flag for next iteration.
            
            # Since door can be tilted, we draw a rotated rectangle around it.
            rect = cv2.minAreaRect( self.contours3[0] )
            
            # Its a tuple ( (centerX, centerY), (height, width), 
            # angle (degrees) ) (all float values).
            rectCenterX, rectCenterY, rectHeight, rectWidth, rectAngle = \
                rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]    
            
            # NOTE: output of minAreaRect is in LH coord. frame, with angles 
            # between -0 and -90.

            # Enforce rectangles that are taller than they are wide, with 
            # the angle measured as negative CCW from the right (x axis).
            if rectHeight < rectWidth:
                rectAngle -= 90
                rectHeight, rectWidth = rectWidth, rectHeight
            
            # Convert from LH frame to RH frame
            rectAngle *= -1.0

#-------------------------------------------------------------------------------

        # Fixing the angle of tilt of rectangle and calculating the depth.
        
        # Initializing the variables to record the distance.
        self.X, self.Y, self.Z = 0.0, 0.0, 0.0
        
        # Sometimes the rectangle drawn is a obscure tilted one. So if the 
        # angle is too tilted, we do not draw it or calculate the depth.
        if abs(rectAngle) > 80.0 and abs(rectAngle) < 100.0:                
            # Define a vector from center of box to center of handle (with 
            # box in default 0-degree position: lying on its right side).
            defaultX = -0.14*rectHeight
            defaultY = -0.31*rectWidth

            # Rotate door into current position.
            handleX = defaultX*np.cos(np.deg2rad(rectAngle)) - \
                      defaultY*np.sin(np.deg2rad(rectAngle))
            handleY = defaultX*np.sin(np.deg2rad(rectAngle)) + \
                      defaultY*np.cos(np.deg2rad(rectAngle))

            # Rotate handle vector back into LH frame (image frame is LH).
            handleY = -1*handleY

            # Deriving the corner vertex from the rotated rectangle. box[0] 
            # will be the vertex with max x and y values (lower right) and 
            # others are numbered in anti-clockwise order from that.
            box = cv2.boxPoints( rect )
            
#-------------------------------------------------------------------------------

            # Calculate the distance from the size of the box.
                        
            # Distance Measurement from tilted rectangle.
            # Distance (in mm) from camera measured using triangle formula, 
            # based on RGB image, as a backup to realsense 
            # (as realsense is not accurate beyond 2000mm).
            X1 = self.F * self.doorWidth / rectWidth      
            X2 = self.F * self.doorHeight / rectHeight
            
            # Taking average of the estimates from door height and width.
            XfromSizeOfBox = (X1+X2)*0.5     
            
#-------------------------------------------------------------------------------

            # Calculate the distance from realsense.

            XfromRealsense = 0.0    # Initializing realsense depth variable.
            
            if self.useRealsense == True:
                # Histogram as a numpy array. Calculating the histogram of the
                # depth image in the region of the combinedMask.
                
                hist = cv2.calcHist( [ self.depthFrame2000mm ], [0], \
                       self.combinedMask, [self.numOfBins], [0, self.maxHistValueRange] )   
                # The histogram shows how many pixels are there at each values 
                # from 0 to 256 (if the bin size is 256 and range is 0 to 256).
                
                # Neglecting the pixels that are in the bins from 0 to 50 (as 
                # those pixels are have distance of 500mm which is outside the 
                # reliable range of realsense) by making their count all 0.
                hist[ : 50 ] = 0            
                
                # This values are in cm as the histogram divides the pixels in
                # bins of 10 values.
                XfromRealsense = np.argmax( hist ) * self.maxHistValueRange * \
                                    1.0 / self.numOfBins    
                # So we take that index of the histogram array which has the 
                # max no. of pixels.
                # This value is in mm now.
            
#-------------------------------------------------------------------------------

            # When inside the range of 500 to 1500mm from the object, we 
            # switch to realsense to calculate the depth X.

            if self.useRealsense == True and self.useRealsenseForDepth == True:
                self.X = XfromRealsense
                
                # Check if the range is outside the valid realsense range, 
                # otherwise reset the flag.
                if XfromRealsense > self.realsenseMaxReliableRange or \
                   XfromRealsense < self.realsenseMinReliableRange:
                    self.useRealsenseForDepth = False
            
            else:
                self.X = XfromSizeOfBox
                
                # Check if the range is inside the valid realsense range, then 
                # make flag true to use realsense distance from next iteration.
                if self.useRealsense == True and \
                   XfromSizeOfBox > self.realsenseMinReliableRange and \
                   XfromSizeOfBox < self.realsenseMaxReliableRange:  
                   self.useRealsenseForDepth = True
                
#-------------------------------------------------------------------------------

        # Within 750mm from the door, no more rectangle display is needed 
        # as the approach will be head-on.

        if self.X > self.minDistToDisplayDepth:     
            # Calculate marker position on handle from tilted rectangle.
            cx = handleX + rectCenterX
            cy = handleY + rectCenterY

            # Calculating the Y and Z distances from the center of handle.
            self.Y = self.X * ( self.width*0.5 - (cx) ) / self.F
            self.Z = self.X * ( self.height*0.5 - (cy) ) / self.F
            
            # Converting x, y, z to meters and then into strings.
            self.X, self.Y, self.Z = self.X/1000.0, self.Y/1000.0, self.Z/1000.0
            
            # Static offsets to reduce the approximation errors (calculated 
            # during testing).
            self.X = self.X - 0.044
            self.Y = self.Y + 0.04
            self.Z = self.Z - 0.07

            #self.Xstr = '{:4.3f}'.format( self.X )
            #self.Ystr = '{:4.3f}'.format( self.Y ) 
            #self.Zstr = '{:4.3f}'.format( self.Z )            
            self.Xstr = '%4.3f'%( self.X )
            self.Ystr = '%4.3f'%( self.Y ) 
            self.Zstr = '%4.3f'%( self.Z )
            
#-------------------------------------------------------------------------------

            # Displaying rectangle around the door and the center of handle.

            # Converting the values of the rectangle into int to display.
            box = np.int0( box )        
            # Drawing the rectangle around the door.
            doorRectColor = (255, 255, 0)
            cv2.drawContours( self.inputColorFrame, [box], 0, doorRectColor, 2 )
            
            cx, cy = int(cx), int(cy)       # Converting to int to display.
            # Drawing the center of the handle.
            cv2.circle( self.inputColorFrame, (cx, cy), 7, (255,0,255), 1 )     
            cv2.circle( self.inputColorFrame, (cx, cy), 2, (255,0,255), 2 )
        
        else:
            # If x,y,z are not measured.
            self.Xstr, self.Ystr, self.Zstr = 'N/A', 'N/A', 'N/A'

#===============================================================================

    def drawCoordinates( self ):
        '''
        Drawing the coordinate axes at the corner of the frames given as input img.
        Also printing some commands and states.
        '''
        if self.inputColorFrame is None:
            return
        
        img = self.inputColorFrame
        h, w = self.height, self.width

        # Making a bar to display text in the image (0.9 times image height).
        barThick = 0.92
        bar = img[ int(h*barThick) : h, 0 : w, : ]
        shade = 0
        bar = bar * 0.5 + shade    # Darkening it. The shade value lightens it.
        
        barH, barW = bar.shape[0], bar.shape[1]
        ox, oy = int(barW*0.98), int(barH*0.8)     # Origin.
        r = int( barH / 8 )
        
        bar = np.asarray( bar, dtype=np.uint8 )  # Convert to uint for drawing axes.
        # Z axis.
        cv2.line( bar, (ox, oy), ( ox, int(barH*0.1) ), (255, 150, 100), 2 )     
        # Y axis.
        cv2.line( bar, (ox, oy), ( int(barW*0.93), oy ), (0, 255, 0), 2 )     
        # X axis.
        cv2.circle( bar, (ox, oy), r, (0, 0, 255), -1 )        
        
        x, y, z, state = self.Xstr, self.Ystr, self.Zstr, self.state
        
        cv2.putText( bar, 'Y: %s' %(y), ( int(ox-barW*0.385), \
        int(oy-barH*0.12) ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2 )
        cv2.putText( bar, 'Z: %s' %(z), ( int(ox-barW*0.22), \
        int(oy-barH*0.12) ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 100), 2 )
        cv2.putText( bar, 'X: %s' %(x), ( int(ox-barW*0.555), \
        int(oy-barH*0.12) ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2 )
        cv2.putText( bar, '%s' %(state), ( int(ox-barW*0.96), \
        int(oy-barH*0.12) ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2 )

        img[ int(h*barThick) : h, 0 : w, : ] = bar     # Add the bar to the image.
        
        self.inputColorFrame = img

#===============================================================================

    def drawRectangle( self, event, x, y, flags, params ):
        '''
        This is a function that is called on mouse callback.
        '''
        
        # ix and iy are initial x and y of the rectangle, fx and fy are final 
        # x and y of the rectangle. cursorX and cursorY are the point of double 
        # click in the window.
        # x, y position recorded once left clicked on image.
        
        if event == cv2.EVENT_LBUTTONDOWN:      
            self.ix, self.iy, self.fx, self.fy, self.drawing = x, y, x, y, True
        # Update the x, y position once mouse is moved after left click.
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing == True:
            self.fx, self.fy = x, y
        # Stop drawing when left click is released.
        elif event == cv2.EVENT_LBUTTONUP:      
            self.drawing = False
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.cursorX, self.cursorY = x, y

#===============================================================================

    # Main function.
    def imageDetection( self ):
        '''
        Function to detect the box using all other functions.
        '''

        # Defining the parameters whose moving average needs to be calculated.
        frameRate = mvAvg( window=30 )     # Calculates average frame rate.

#-------------------------------------------------------------------------------

        # Initializing the cameras.

        if self.useRealsense == True:
            # Defining the video recorders.
            colRec = videoRecorder( fps=60 )
            depRec = videoRecorder( fps=60 )
            
            # Record raw videos as well when recordRawVideoFramsToo flag is True.
            if self.recordRawVideoFramsToo:     
                rawRec = videoRecorder( fps=60 )
                
            # Defining display windows and mousecallback functions.
            delay = 1     # The delay will be used with the cv2.waitKey().
            #cv2.namedWindow( 'Depth' )
            cv2.namedWindow( 'Color' )
            
            # Drawing bounding boxes on the live frames.
            cv2.setMouseCallback( 'Color', self.drawRectangle )
            
        else:
            # Defining the video recorders.
            colRec = videoRecorder( self.camera )
            
            # Record raw videos as well when recordRawVideoFramsToo flag is True.
            if self.recordRawVideoFramsToo:     
                rawRec = videoRecorder( self.camera )
            
            delay = 1     # The delay will be used with the cv2.waitKey().
            # Defining display windows and mousecallback functions.
            cv2.namedWindow( 'Color' )
            # Drawing bounding boxes on the live frames.
            cv2.setMouseCallback( 'Color', self.drawRectangle )
            
        print('\n')

#-------------------------------------------------------------------------------

        # Main loop.

        while self.loopCondition:    # esc key for exit.
            
            self.loopCondition = self.key & 0xFF != 27    # Updating the loopCondition.
            
            startTime = time.time()

#-------------------------------------------------------------------------------

            # Reading the input frames.

            self.readFrames()
            
            # We are never touching originalColorFrame, always working with a copy.
            self.inputColorFrame = np.array( self.originalColorFrame )      
            self.height, self.width, self.channel = self.inputColorFrame.shape

#-------------------------------------------------------------------------------

            # Remove the clutter due to the end effector.

            self.removeEndEffectorClutter()

#-------------------------------------------------------------------------------

            # Pre-pocessing the frames to get a good contour detection.

            self.preProcess()
            
#-------------------------------------------------------------------------------

            # Drawing rectangles and measuring distances when svm predicts positive.

            self.findDistance()

#-------------------------------------------------------------------------------

            # Determining the state.

            self.state = ''     # Will be obtained from ROS after interfacing drone.
            
            # This part of the code will be written while interfacing with the
            # actual drone.

#-------------------------------------------------------------------------------

            # Draw axes and display x, y, z and state values.

            self.drawCoordinates()
            
#-------------------------------------------------------------------------------

            # Drawing the cross hairs in the frames.

            cv2.drawMarker( self.inputColorFrame, ( int(self.width/2), \
              int(self.height/2) ), (0, 0, 255), markerType=cv2.MARKER_CROSS, \
                  markerSize=50, thickness=4 )
            cv2.drawMarker( self.inputColorFrame, ( int(self.width/2), \
              int(self.height/2) ), (0, 0, 255), markerType=cv2.MARKER_CROSS, \
                  markerSize=100, thickness=1 )
            
            # Drawing the rectangle over the display window by mouse click.
            # Rectangle drawn by mouseclick.
            cv2.rectangle( self.inputColorFrame, (self.ix, self.iy), \
                (self.fx, self.fy), (0, 255, 0), 2 )    

            # Displaying frames.
            cv2.imshow( 'Color', self.inputColorFrame )   # Display color frame.
            #cv2.imshow( 'modifiedRGBframe', self.modifiedInputFrame )
            
            if self.useRealsense == True:    # Display depth frame.
                #cv2.imshow( 'Depth', self.depthFrame200cm )  
                pass
            
#-------------------------------------------------------------------------------

            # Create training data for hog and svm.

            # Data will not be recorded if the './frames' folder is not created 
            # (in the directory of the script) before the code is run and the 
            # self.recordTrainingData flag is not True.
            # The regions where the rectangles are drawn are saved as images in 
            # './frames' folder. These can be used as training images.
            self.recordTrainingData = True if self.key & 0xFF == ord('t') \
                else self.recordTrainingData
            
            if self.recordTrainingData and self.x != -1:
                cv2.rectangle( self.modifiedInputFrame, \
                (self.x-15, self.y-15), (self.x+self.w+15, self.y+self.h+15), \
                (255,255,255), 2 )    # Only drawing the filtered rectangles.
                
                trainImg = self.modifiedInputFrame[ 
                self.y-15 : self.y+self.h+15, self.x-15:self.x+self.w+15, : ]
                
                #cv2.imwrite( os.path.join( self.filePath, 'frames/image_{}_{:0.3f}.png'.format( \
                    #time.strftime('_%d%b%Y_%H_%M_%S'), time.time() ), trainImg ) )
                cv2.imwrite( os.path.join( self.filePath, 'frames/image_%s_%0.3f.png' %( \
                    time.strftime('_%d%b%Y_%H_%M_%S'), time.time() ), trainImg ) )
                
#-------------------------------------------------------------------------------

            # Operations on keypress.

            self.key = cv2.waitKey(delay)
            
            #print( '\rFrame rate: {:0.3f} fps'.format( frameRate.calc( 1 / \
                #(time.time() - startTime) ) ), end='' )
            print( '\rFrame rate: %0.3f fps' %( frameRate.calc( 1 / \
                (time.time() - startTime) ) ) ),
            delay = 0 if self.key & 0xFF == ord(' ') else delay  # 'space': pause.
            delay = 1 if self.key & 0xFF == 13 else delay    # 'enter': play.
            
            # 'a' key saves current video frame as an image in current directory.
            if self.key & 0xFF == ord('a'):
                cv2.imwrite( os.path.join( self.filePath, 'colorImage' + \
                    time.strftime('_%d%b%Y_%H_%M_%S') + \
                    '.png' ), self.inputColorFrame )

                # Also save raw image if the recordRawVideoFramsToo flag is True.
                if self.recordRawVideoFramsToo:     
                    cv2.imwrite( os.path.join( self.filePath, 'rawImage' + \
                        time.strftime('_%d%b%Y_%H_%M_%S') + \
                        '.png' ), self.originalColorFrame )
                
                # Also save the depth image if the realsense is active.
                if self.useRealsense:
                    cv2.imwrite( os.path.join( self.filePath, 'depthImage' + \
                        time.strftime('_%d%b%Y_%H_%M_%S') + \
                            '.png' ), self.depthFrame200cm )

            # 'm' resets global variables and calls function to do some tasks 
            # with the rectangle drawn by mouse callback.
            if self.key & 0xFF == ord('m'):
                self.ix, self.iy, self.fx, self.fy = -1, -1, -1, -1
                pass

#-------------------------------------------------------------------------------

            # Recording the frames.

            # If 'r' is pressed once, recording will start and if 's' is 
            # pressed once, recording will stop.
            colRec.record( os.path.join( self.filePath, 'color' ), \
                self.inputColorFrame, recordFrame=(self.key & 0xFF) )  
            
            # Also record the raw video using the original video frames if the 
            # recordRawVideoFramsToo flag is True.
            if self.recordRawVideoFramsToo:     
                rawRec.record( os.path.join( self.filePath, 'raw' ), \
                    self.originalColorFrame, recordFrame=(self.key & 0xFF) )

            # Also record the depth video if the realsense is active.
            if self.useRealsense:     
                depRec.record( os.path.join( self.filePath, 'depth' ), \
                    self.depthFrame200cm, recordFrame=(self.key & 0xFF) )

#-------------------------------------------------------------------------------

        # Shutting down everyting.

        cv2.destroyAllWindows()
        print('\n')

        if self.useRealsense:
            self.camera.stop()    # Stop camera and service.
            self.serv.stop()
        else:
            self.camera.release()

#===============================================================================

if __name__ == '__main__':
    bd = BoxDetector( useRealsense=False, recordRawVideoFramsToo=False, \
                       useRos=False, filePath='./' )
    bd.imageDetection()

