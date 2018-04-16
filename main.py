# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os, shutil
import dataPrep

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================


def reduce_dimensions(X):
    '''
    Reduce the dimensionality of X with different reducers.

    Return a sequence of tuples containing:
        (title, x coordinates, y coordinates)
    for each reducer.
    '''

    # Principal Component Analysis (PCA) is a linear reduction model
    # that identifies the components of the data with the largest
    # variance.
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2)
    X_r = reducer.fit_transform(X)
    yield 'PCA', X_r[:, 0], X_r[:, 1]

    # Independent Component Analysis (ICA) decomposes a signal by
    # identifying the independent contributing sources.
    from sklearn.decomposition import FastICA
    reducer = FastICA(n_components=2)
    X_r = reducer.fit_transform(X)
    yield 'ICA', X_r[:, 0], X_r[:, 1]

    # t-distributed Stochastic Neighbor Embedding (t-SNE) is a
    # non-linear reduction model. It operates best on data with a low
    # number of attributes (<50) and is often preceded by a linear
    # reduction model such as PCA.
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2)
    X_r = reducer.fit_transform(X)
    yield 't-SNE', X_r[:, 0], X_r[:, 1]


def evaluate_learners(trainData, testData):
    '''
    Run multiple times with different learners to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, predicted classes)
    for each learner.
    '''

    from sklearn.cluster import (MeanShift, MiniBatchKMeans,
                                 SpectralClustering, AgglomerativeClustering)

    learner = MeanShift(
        # Let the learner use its own heuristic for determining the
        # number of clusters to create
        bandwidth=None
    )
    y = learner.fit_predict(trainData)
    yield 'Mean Shift clusters train', y, 0
    y = learner.predict(testData)
    yield 'Mean Shift clusters test', y, 1

    learner = MiniBatchKMeans(n_clusters=2)
    y = learner.fit_predict(trainData)
    yield 'K Means clusters train', y, 0
    y = learner.predict(testData)
    yield 'K Means clusters test', y, 1

    learner = SpectralClustering(n_clusters=2)
    y = learner.fit_predict(trainData)
    yield 'Spectral clusters train', y, 0

    learner = AgglomerativeClustering(n_clusters=2)
    y = learner.fit_predict(trainData)
    yield 'Agglo clusters (N=2) train', y, 0

    learner = AgglomerativeClustering(n_clusters=5)
    y = learner.fit_predict(trainData)
    yield 'Agglo clusters (N=5) train', y, 0


# =====================================================================


def plot(trainDataReduced, testDataReduced, trainPredictions, testPredictions):
    '''
    Create a plot comparing multiple learners.

    `Xs` is a list of tuples containing:
        (title, x coord, y coord)
    
    `predictions` is a list of tuples containing
        (title, predicted classes)

    All the elements will be plotted against each other in a
    two-dimensional grid.
    '''

    # We will use subplots to display the results in a grid
    nrows = len(trainDataReduced)
    ncols = len(trainPredictions) + len(testPredictions)

    fig = plt.figure(figsize=(20, 8))
    fig.canvas.set_window_title("Clustering results")

    # Show each element in the plots returned from plt.subplots()
    
    for row, (row_label, X_x, X_y) in enumerate(trainDataReduced):
        for col, (col_label, y_pred, type) in enumerate(trainPredictions):
            ax = plt.subplot(nrows, ncols, row * ncols + col + 1)
            if row == 0:
                plt.title(col_label)
            if col == 0:
                plt.ylabel(row_label)

            # Plot the decomposed input data and use the predicted
            # cluster index as the value in a color map.
            plt.scatter(X_x, X_y, c=y_pred.astype(np.float), cmap='prism', alpha=0.5)
            
            # Set the axis tick formatter to reduce the number of ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    for row, (row_label, X_x, X_y) in enumerate(testDataReduced):
        for col, (col_label, y_pred, type) in enumerate(testPredictions):
            ax = plt.subplot(nrows, ncols, row * ncols + col + len(trainPredictions) + 1)
            if row == 0:
                plt.title(col_label)
            if col == 0:
                plt.ylabel(row_label)

            # Plot the decomposed input data and use the predicted
            # cluster index as the value in a color map.
            plt.scatter(X_x, X_y, c=y_pred.astype(np.float), cmap='prism', alpha=0.5)
            
            # Set the axis tick formatter to reduce the number of ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Let matplotlib handle the subplot layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('plot.png')

    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plot.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    #from io import BytesIO
    #img_stream = BytesIO()
    #plt.savefig(img_stream, fmt='png')
    #img_bytes = img_stream.getvalue()
    #print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()


# =====================================================================

def copyFilesByClasses(predictions, fileNames):
    dataPath = "data\\"
    try:
        shutil.rmtree("result")
    except Exception:
        pass
    for (method, y_pred, type) in predictions:
        dstPath = "result\\" + method + "\\"
        for i in range(len(y_pred)):
            src = dataPath + fileNames[i]
            dst = dstPath + str(y_pred[i]) + "\\"
            if not os.path.exists(dst):
                os.makedirs(dst)
            shutil.copyfile(src, dst + fileNames[i])

if __name__ == '__main__':
    # Get data set and image names
    from sklearn.model_selection import train_test_split
    print("Preparing the data set and file names")
    dataSet, fileNames = dataPrep.getImageDataSet()
    trainData, testData, trainFiles, testFiles = train_test_split(dataSet, fileNames, test_size = 0.25)

    # Run multiple dimensionality reduction algorithms on the data
    print("Reducing dimensionality")
    trainDataReduced = list(reduce_dimensions(trainData))
    testDataReduced = list(reduce_dimensions(testData))

    # Evaluate multiple clustering learners on the data
    print("Evaluating clustering learners")
    predictions = list(evaluate_learners(trainData, testData))
    trainPredictions = []
    testPredictions = []
    for prediction in predictions:
        if(prediction[2] == 0):
            trainPredictions.append(prediction)
        else:
            testPredictions.append(prediction)

    # Copy images to separated folders by the clustering methods and their classes
    print("Copying images")
    copyFilesByClasses(predictions, fileNames)

    # Display the results
    print("Plotting the results")
    plot(trainDataReduced, testDataReduced, trainPredictions, testPredictions)