import numpy as np
import scipy.ndimage as ndimage

class State:

    def stateByAddingScreen(self, screen):
        screen = ndimage.zoom(screen, 0.5)
        newState = State()
        # Consider resizing here vs in TF to save memory (??) and be able to store more replaymemory
        if hasattr(self, 'screen'):
            newState.screens = np.append(screen, self.screens[:, :, :3], axis=2)
        else:
            screen = np.reshape(screen, (screen.shape[0], screen.shape[1]))
            newState.screens = np.stack((screen, screen, screen, screen), axis=2)
        return newState
