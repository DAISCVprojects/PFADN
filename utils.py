import numpy as np

def I2channels( I ):
    """Naively demosaic a PFA image to I0, I45, I90, I135. The output images are half the with and height of the input

    Args:
        I (numpy array): Input mosaiced image

    Returns:
        I0, I45, I90, I135: Demosaiced camera channels (half size)
    """    
    assert I.dtype == np.float32 or I.dtype == np.float64
    assert I.ndim == 2

    I90 = I[::2,::2]
    I0 = I[1::2,1::2]
    I45 = I[::2,1::2]
    I135 = I[1::2,::2]
    return I0, I45, I90, I135


def channels2stokes( I0, I45, I90, I135 ):
    """Computes Stokes vector from PFA camera channels

    Args:
        I0 ([type]): Channel I0
        I45 ([type]): Channel I1
        I90 ([type]): Channel I2
        I135 ([type]): Channel I3

    Returns:
        S0,S1,S2: Stokes vector
    """    

    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135
    return S0, S1, S2 


def aolp( S0, S1, S2 ):
    """ Computes Angle of Linear Polarization from Stokes parameters
    """    
    return 0.5 * np.arctan2(S2,S1)


def dolp( S0, S1, S2 ):
    """ Computes Degree of Linear Polarization from Stokes parameters
    """    
    return np.sqrt( S1**2 + S2**2 ) / S0


def stokes2channels( S0, S1, S2 ):
    """Computes (ideal) PFA camera channels from Stokes vector

    Args:
        S0 (numpy array): Stokes S0
        S1 (numpy array): Stokes S1
        S2 (numpy array): Stokes S2

    Returns:
        I0, I45, I90, I135: Camera channels
    """        

    I0   = 0.5 * S0 * (1.0 + dolp * (np.cos( 2*aolp - 2*0 ) ) )
    I45  = 0.5 * S0 * (1.0 + dolp * (np.cos( 2*aolp - 2*np.pi*0.25 ) ) )
    I90  = 0.5 * S0 * (1.0 + dolp * (np.cos( 2*aolp - np.pi ) ) )
    I135 = 0.5 * S0 * (1.0 + dolp * (np.cos( 2*aolp - 3*np.pi*0.5 ) ) )

    return I0, I45, I90, I135


def optimize_channels( I0, I45, I90, I135, good_px_thresh=0.05 ):
    """Optimizes channels to enforce I0+I90=I45+I135 constraint

    Args:
        I0 ([type]): Channel I0
        I45 ([type]): Channel I1
        I90 ([type]): Channel I2
        I135 ([type]): Channel I3
        good_px_thresh (float, optional): Threshold to consider a channel value "reliable". Defaults to 0.05.

    Returns:
        Optimized I0, I45, I90, I135 + reliable pixels boolean mask
    """    
    Icube = np.copy( np.array( [I0, I45, I90, I135 ] ) )

    good_px = (Icube>good_px_thresh).astype(np.uint8) * (Icube<1-good_px_thresh).astype(np.uint8)
    n_good_px = np.sum( good_px, axis=0 )

    good_mask = (n_good_px==4)
    
    Icube[0,:,:] = 0.75 * I0 + 0.25 * I45 - 0.25 * I90 + 0.25 * I135
    Icube[1,:,:] = 0.25 * I0 + 0.75 * I45 + 0.25 * I90 - 0.25 * I135
    Icube[2,:,:] = 0.25 * I0 - 0.25 * I45 + 0.75 * I90 + 0.25 * I135
    Icube[3,:,:] = 0.25 * I0 - 0.25 * I45 + 0.25 * I90 + 0.75 * I135

    return Icube[0,...], Icube[1,...], Icube[2,...], Icube[3,...], good_mask


def render_tiled( I0, I45, I90, I135 ):
    assert I0.shape[0]== I135.shape[0]
    assert I0.shape[1]== I90.shape[1]

    Itiled = np.zeros( (I0.shape[0]+I45.shape[0], I0.shape[1]+I90.shape[1]), I0.dtype )

    Itiled[:I0.shape[0],:I0.shape[1]] = I0
    Itiled[:I0.shape[0],I0.shape[1]:] = I45
    Itiled[I0.shape[0]:,:I90.shape[1]:] = I90
    Itiled[I0.shape[0]:,I135.shape[1]:] = I135
    return (Itiled*255.0).astype(np.uint8)


def render_tiled_polar( S0, dolp, aolp, good_mask = None ):
    import cv2 as cv

    assert S0.shape == dolp.shape
    assert dolp.shape == aolp.shape

    S0_col = cv.cvtColor( np.clip(S0*140.0,0,255).astype(np.uint8), cv.COLOR_GRAY2RGB )
    Dolp_col = cv.applyColorMap( (dolp*255.0).astype(np.uint8), cv.COLORMAP_JET )
    Aolp_col = cv.applyColorMap( ( (aolp+np.pi*0.5)/np.pi*255.0).astype(np.uint8), cv.COLORMAP_HSV )

    if not good_mask is None:
        Dolp_col[ np.logical_not(good_mask), :] = 255
        Aolp_col[ np.logical_not(good_mask), :] = 255

    Itiled = np.zeros( (S0_col.shape[0], S0_col.shape[1]*3, 3), dtype=np.uint8 )
    Itiled[:, :S0_col.shape[1], : ] = S0_col
    Itiled[:, S0_col.shape[1]:(S0_col.shape[1]+Dolp_col.shape[1]), : ] = Dolp_col
    Itiled[:, (S0_col.shape[1]+Dolp_col.shape[1]):, : ] = Aolp_col

    return Itiled


def angleLoss2_deg( angle_pred, angle_true ):

    # apply padding
    pad = 10
    angle_pred = angle_pred[pad:-pad, pad:-pad]
    angle_true = angle_true[pad:-pad, pad:-pad]
   
    angle_pred = angle_pred + np.pi*0.5  # 0..pi
    angle_true = angle_true + np.pi*0.5  # 0..pi
    
    a1 = np.abs( angle_pred-angle_true )
    a2 = np.pi - a1
    
    return np.nanmean( np.minimum( a1, a2 )*180.0 / np.pi )

