import numpy as np
import scipy.ndimage as ndimage
import png

class State:

    def stateByAddingScreen(self, screen):
        screen = ndimage.zoom(screen, (0.5, 0.5, 1.0))
        newState = State()
        # Consider resizing here vs in TF to save memory (??) and be able to store more replaymemory
        if hasattr(self, 'screens'):
            newState.screens = np.append(screen, self.screens[:, :, :3], axis=2)
        else:
            screen = np.reshape(screen, (screen.shape[0], screen.shape[1]))
            newState.screens = np.stack((screen, screen, screen, screen), axis=2)
        return newState
    
    def saveStateAsPNGs(self, basefilename):
        for i in range(4):
            pngfile = open(basefilename + ('-%d.png' % i), 'wb')
            screen = self.screens[:,:,i]
            print(screen.shape)
            pngWriter = png.Writer(screen.shape[1], screen.shape[0], greyscale=True)
            pngWriter.write(pngfile, screen)
            pngfile.close()
