import numpy as np
import png

class State:

    def stateByAddingScreen(self, screen, frameNumber):
        #self.saveScreenAsPNG('screen', screen.getPixels(), frameNumber)
        
        newState = State()
        if hasattr(self, 'screens'):
            newState.screens = self.screens[:3]
            newState.screens.insert(0, screen)
        else:
            newState.screens = [screen, screen, screen, screen]
        return newState
    
    def getScreens(self):
        s = []
        for i in range(4):
            s.append(self.screens[i].getPixels())
        return np.concatenate(s, axis=2)
    
    def saveScreenAsPNG(self, basefilename, screen, frameNumber):
        pngfile = open(basefilename + ('-%08d.png' % frameNumber), 'wb')
        pngWriter = png.Writer(screen.shape[1], screen.shape[0], greyscale=True)
        pngWriter.write(pngfile, screen)
        pngfile.close()
