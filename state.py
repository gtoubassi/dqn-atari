import numpy as np
import scipy.ndimage as ndimage
import png

class State:

    def stateByAddingScreen(self, screen, frameNumber):
        screen = np.dot(screen, np.array([0.2126, 0.7152, 0.0722])).astype(np.uint8)
        screen = ndimage.zoom(screen, (0.5, 0.5))
        screen.resize((105, 80, 1))
        #self.saveScreenAsPNG('screen', screen, frameNumber)
        newState = State()
        if hasattr(self, 'screens'):
            newState.screens = np.append(screen, self.screens[:, :, :3], axis=2)
        else:
            screen = np.reshape(screen, (screen.shape[0], screen.shape[1]))
            newState.screens = np.stack((screen, screen, screen, screen), axis=2)
        return newState
    
    def saveScreenAsPNG(self, basefilename, screen, frameNumber):
        pngfile = open(basefilename + ('-%08d.png' % frameNumber), 'wb')
        pngWriter = png.Writer(screen.shape[1], screen.shape[0], greyscale=True)
        pngWriter.write(pngfile, screen)
        pngfile.close()
    
    def saveStateAsPNGs(self, basefilename):
        for i in range(4):
            pngfile = open(basefilename + ('-%d.png' % i), 'wb')
            screen = self.screens[:,:,i]
            pngWriter = png.Writer(screen.shape[1], screen.shape[0], greyscale=True)
            pngWriter.write(pngfile, screen)
            pngfile.close()
