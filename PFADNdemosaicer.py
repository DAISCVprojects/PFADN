import numpy as np
from PFADN_github.PFADN import create_model
import cv2 as cv
from utils import I2channels, channels2stokes, optimize_channels, dolp 
import h5py

def image_to_tiles( I, KSIZE ):
    W = I.shape[1]
    H = I.shape[0]
    assert W%KSIZE == 0
    assert H%KSIZE == 0
    
    If = np.reshape( I, (H//KSIZE,KSIZE,W//KSIZE,KSIZE) )
    If = np.swapaxes(If,1,2)
    tiled_shape = If.shape
    If = np.reshape( If, (tiled_shape[0]*tiled_shape[1],tiled_shape[2],tiled_shape[3]))
    return If, tiled_shape



def tiles_to_image( If, tiled_shape, image_shape ):
    If = np.reshape( If, tiled_shape)
    If = np.swapaxes(If,2,1)
    Ik = np.reshape( If, image_shape )
    return Ik



class PFADNdemosaicer():
    
    def __init__(self, weightsfile, KSIZE=128 ):
        
        self.KSIZE = KSIZE
        self.model = create_model(KSIZE,KSIZE)
        self.model.load_weights(weightsfile)
        
        self.tilemask = np.zeros( (KSIZE,KSIZE), dtype=np.float32 )
        
        # Yes, 7 is hardcoded but it works well
        pad = 13
        self.tilemask[pad:-pad,pad:-pad] = 1.0
        
        
    def _demosaic_helper( self, I ):
        tiles, tiled_shape = image_to_tiles(I, self.KSIZE)


        tiles_demo = self.model.predict( tiles[...,np.newaxis] )

        tiles_aolp = np.arctan2(tiles_demo[...,1], tiles_demo[...,0])*0.5
        tiles_I = tiles_demo[...,2]

        Idemo = tiles_to_image( tiles_I, tiled_shape, I.shape)
        aolp = tiles_to_image( tiles_aolp, tiled_shape, I.shape )

        return Idemo, aolp
    
    
        
    def demosaic( self, I, optimize=True ):

        if I.dtype==np.uint8:
            I=I.astype(np.float32)/255.0

        I0, I45, I90, I135 = I2channels(I)
        if optimize:
            I0, I45, I90, I135, good_mask = optimize_channels(I0,I45,I90,I135)
        S0, S1, S2 = channels2stokes( I0, I45, I90, I135 )

        PAD_W = self.KSIZE - I.shape[1]%self.KSIZE
        PAD_H = self.KSIZE - I.shape[0]%self.KSIZE
        
        # Pad to ensure that the image size is a multiple of KSIZE
        I = np.pad(I, ((0,PAD_H),(0,PAD_W)) ) 
        
        # Demosaic the original image
        I1, aolp1 = self._demosaic_helper(I)
        
        # Demosaic the same image shifted of half KSIZE to fix discontinuities at tile borders
        Ir = np.roll(I,(-self.KSIZE//2,-self.KSIZE//2), (0,1))
        I2, aolp2 = self._demosaic_helper(Ir)
        
        I2 = np.roll(I2,(self.KSIZE//2,self.KSIZE//2), (0,1))
        aolp2 = np.roll(aolp2,(self.KSIZE//2,self.KSIZE//2), (0,1))
        
        full_tilemask = np.tile(self.tilemask, (I.shape[0]//self.KSIZE*I.shape[1]//self.KSIZE,1,1))
        full_tilemask = tiles_to_image(full_tilemask,(I.shape[0]//self.KSIZE,I.shape[1]//self.KSIZE,self.KSIZE,self.KSIZE), I.shape )
        
        # .. mix them to remove the artefacts!
        Id = I1*full_tilemask + I2*(1-full_tilemask)
        aolp = aolp1*full_tilemask + aolp2*(1-full_tilemask)

        Id = Id[:-PAD_H,:-PAD_W]
        aolp = -aolp[:-PAD_H,:-PAD_W]

        # remove padding and return        
        d = dolp(S0,S1,S2)
        d = cv.resize( d, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC )
        return Id, d, aolp
        
        
        
        