import numpy as np
import scipy.ndimage as ndimage
import png

class State:

    def stateByAddingScreen(self, screen, frameNumber):
        screen = np.dot(screen, np.array([.299, .587, .114])).astype(np.uint8)
        screen = ndimage.zoom(screen, (0.5, 0.5))
        screen.resize((105, 80, 1))
        #self.saveScreenAsPNG('screen', screen, frameNumber)
        newState = State()
        if hasattr(self, 'screens'):
            newState.screens = self.screens[:3]
            newState.screens.insert(0, screen)
        else:
            newState.screens = [screen, screen, screen, screen]
        return newState
    
    def getScreens(self):
        return np.concatenate(self.screens, axis=2)
    
    def saveScreenAsPNG(self, basefilename, screen, frameNumber):
        pngfile = open(basefilename + ('-%08d.png' % frameNumber), 'wb')
        pngWriter = png.Writer(screen.shape[1], screen.shape[0], greyscale=True)
        pngWriter.write(pngfile, screen)
        pngfile.close()
