import numpy as np
import scipy.ndimage as ndimage
import blosc
import png

class State:

    useCompression = False

    @staticmethod
    def setup(args):
        State.useCompression = args.compress_replay

    def stateByAddingScreen(self, screen, frameNumber):
        screen = np.dot(screen, np.array([.299, .587, .114])).astype(np.uint8)
        screen = ndimage.zoom(screen, (0.4, 0.525))
        screen.resize((84, 84, 1))
        #self.saveScreenAsPNG('screen', screen, frameNumber)
        
        if State.useCompression:
            screen = blosc.compress(np.reshape(screen, 84 * 84).tobytes(), typesize=1)

        newState = State()
        if hasattr(self, 'screens'):
            newState.screens = self.screens[:3]
            newState.screens.insert(0, screen)
        else:
            newState.screens = [screen, screen, screen, screen]
        return newState
    
    def getScreens(self):
        if State.useCompression:
            s = []
            for i in range(4):
                s.append(np.reshape(np.fromstring(blosc.decompress(self.screens[i]), dtype=np.uint8), (84, 84, 1)))
        else:
            s = self.screens
        return np.concatenate(s, axis=2)
    
    def saveScreenAsPNG(self, basefilename, screen, frameNumber):
        pngfile = open(basefilename + ('-%08d.png' % frameNumber), 'wb')
        pngWriter = png.Writer(screen.shape[1], screen.shape[0], greyscale=True)
        pngWriter.write(pngfile, screen)
        pngfile.close()
