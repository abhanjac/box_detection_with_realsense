#!/usr/bin/env python

import cv2, numpy as np, time, os, h5py, pickle
from tqdm import tqdm
from skimage import feature, exposure
from sklearn.svm import SVC

# This script can be used for recording data for the svm, training the svm and also testing the svm.
# The dataset has 117 positive examples and 296 negative examples.
recordData = True
trainSVM = True
testSVM = True


if recordData == True:     # Data recording phase.
    positiveFileLocation = './svm_data/train/svm_pos'
    negativeFileLocation = './svm_data/train/svm_neg'
    fileLocList = [ positiveFileLocation, negativeFileLocation ]
    featureFile = './svm_data/train/featureFile.hdf5'

    posFeatures, negFeatures = [], []

    for fl in fileLocList:
        listOfFiles = os.listdir( fl )  # Accessing all the file locations.
        print( 'Number of files in {}: {}'.format( fl, len(listOfFiles) ) )
        
        for i in tqdm( listOfFiles ):
            img = cv2.imread( os.path.join( fl, i ) )
            h, w, c = img.shape
            imgWithCropBorder = img[8:h-8, 8:w-8, :]    # Removing the outer white border.
            imgSmall = cv2.resize( imgWithCropBorder, (60, 72), cv2.INTER_AREA )    # Shrinking the images.
            gray = cv2.cvtColor( imgSmall, cv2.COLOR_BGR2GRAY )
            
            (hog, hogImg) = feature.hog( gray, orientations=9, pixels_per_cell=(12,12), cells_per_block=(5,5), block_norm='L2-Hys', visualise=True, transform_sqrt=True )   # Extracting the hog features.
            
            if fl == positiveFileLocation:
                posFeatures.append( hog )   # Storing the positive hog features.
            elif fl == negativeFileLocation:
                negFeatures.append( hog )   # Storing the negative hog features.
                
            hogImg = exposure.rescale_intensity( hogImg, out_range=(0,255) )
            
            hogImg = hogImg.astype('uint8')
            cv2.imshow('image', hogImg)
            cv2.imshow('original', imgSmall)
            cv2.waitKey(5)

    cv2.destroyAllWindows()

    with h5py.File(featureFile, 'w') as f:
        dset1 = f.create_dataset( 'posFeatures', ( len(posFeatures), len(posFeatures[0]) ), dtype='float' )
        dset1[ 0:len(posFeatures) ] = posFeatures
        dset2 = f.create_dataset( 'negFeatures', ( len(negFeatures), len(negFeatures[0]) ), dtype='float' )
        dset2[ 0:len(negFeatures) ] = negFeatures
        
    ## Cross checking whether the data is stored properly or not.
    #with h5py.File(featureFile, 'r') as f:
        #print( f.keys() )
        #print(posFeatures[0], '\n\n')
        #print( f['posFeatures'][0] )
        #print(negFeatures[0], '\n\n')
        #print( f['negFeatures'][0] )


if trainSVM == True:       # SVM training phase.
    featureFile = './svm_data/train/featureFile.hdf5'
    modelFile = './model.p'
    
    with h5py.File(featureFile, 'r') as f:      # Reading data.
        #listOfNames = [ name for name in f ]
        #print(listOfNames)
        posFeatures = f['posFeatures'][:, :]
        posLabels = np.ones( posFeatures.shape[0] )     # Positive labels are 1.
        negFeatures = f['negFeatures'][:, :]
        negLabels = np.zeros( negFeatures.shape[0] )    # Negative labels are 0.
        features = np.vstack( ( posFeatures, negFeatures ) )    # Concatenating the positive and negative datasets and labels.
        labels = np.hstack( ( posLabels, negLabels ) )
        #print( features.shape, labels.shape )
        
    print( 'Positive set: {}, Negative set: {}'.format( posFeatures.shape, negFeatures.shape ) )
    model = SVC( kernel="linear", C=10, probability=True, random_state=51 )     # C < 1 will allow soft margins. C > 1 produces hard margins. But higher the C value above 1 will make model more prone to overfitting.
    model.fit( features, labels )
    f = open( modelFile, 'wb' )
    pickle.dump( model, f, protocol=1 )     # The protocol = 1 will make the file compatible in both python 2 and 3.
    f.close()
    
    
if testSVM == True:        # SVM testing phase.
    modelFile = './model.p'
    f = open( modelFile, 'rb' )
    model = pickle.load( f )
    
    testFeatureFileLoc = './svm_data/test/neg'
    #testFeatureFileLoc = './svm_data/train/svm_pos'
    listOfTestFiles = os.listdir( testFeatureFileLoc )
    
    for idx, i in tqdm( enumerate( listOfTestFiles ) ):
        img = cv2.imread( os.path.join( testFeatureFileLoc, i ) )
        h, w, c = img.shape
        imgWithCropBorder = img[8:h-8, 8:w-8, :]    # Removing the outer white border.
        imgSmall = cv2.resize( imgWithCropBorder, (60, 72), cv2.INTER_AREA )    # Shrinking the images.
        gray = cv2.cvtColor( imgSmall, cv2.COLOR_BGR2GRAY )
        
        (hog, hogImg) = feature.hog( gray, orientations=9, pixels_per_cell=(12,12), cells_per_block=(5,5), block_norm='L2-Hys', visualise=True, transform_sqrt=True )   # Extracting the hog features.
        hogImg = exposure.rescale_intensity( hogImg, out_range=(0,255) )
        
        hog = np.array( [hog] )     # Converting the (450,) shape into (1, 450) shape to feed it into the model.
        #print(hog.shape)
        
        outputProb = model.predict_proba( hog )     # Predicting the probability. The output will be an array which has the probability of object 0 at location 0 and 1 at location 1.
        outputLabel = model.predict( hog )      # Predicting the label. A list with the correct index in it is returned.
        print( 'Idx: {}, Probability: {}, Label: {}'.format( idx, outputProb, outputLabel[0] ) )
        
        hogImg = hogImg.astype('uint8')
        cv2.imshow('image', hogImg)
        cv2.imshow('original', imgSmall)
        
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    
    
