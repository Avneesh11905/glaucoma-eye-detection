import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.filters import gabor

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Green channel + denoise + CLAHE + sharpen"""
    green    = img[:, :, 1]
    denoised = cv2.fastNlMeansDenoising(green, h=10)
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    kernel   = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(enhanced, -1, kernel)

def crop_optic_disc(img: np.ndarray, size: int = 80) -> np.ndarray:
    """Crop the optic disc region from fundus image."""
    h, w   = img.shape
    cx, cy = int(w * 0.60), int(h * 0.50)
    half   = size // 2
    x1, x2 = max(0, cx-half), min(w, cx+half)
    y1, y2 = max(0, cy-half), min(h, cy+half)
    crop   = img[y1:y2, x1:x2]
    
    if crop.shape[0] > 0 and crop.shape[1] > 0:
        return cv2.resize(crop, (size, size))
    return cv2.resize(img, (size, size))

def glcm_features(u8: np.ndarray) -> list:
    """GLCM texture: energy, contrast, homogeneity, correlation, dissimilarity"""
    r = u8 // 4
    g = graycomatrix(r, distances=[1,2,3],
                     angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                     levels=64, symmetric=True, normed=True)
    feats = []
    for p in ['energy','contrast','homogeneity','correlation','dissimilarity']:
        feats += [graycoprops(g, p).mean(), graycoprops(g, p).std()]
    return feats

def entropy_features(u8: np.ndarray) -> list:
    """Shannon, Renyi, Kapur, Yager entropy"""
    flat = u8.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0,256))
    px = hist / (len(flat) + 1e-10)
    px = px[px > 0]
    shan = -np.sum(px * np.log2(px + 1e-10))
    reny = (1/(1-2)) * np.log2(np.sum(px**2) + 1e-10)
    kapu = (1/1.5) * np.log2((np.sum(px**0.5)/(np.sum(px**2)+1e-10))+1e-10)
    yage = 1 - np.sum(np.abs(2*px-1)) / len(flat)
    return [shan, reny, kapu, yage]

def lbp_features(u8: np.ndarray) -> list:
    """Local binary pattern histogram — captures micro-textures"""
    lbp = local_binary_pattern(u8, P=24, R=3, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0,26), density=True)
    return hist.tolist()

def hog_features(u8: np.ndarray) -> list:
    """HOG gradient features — captures optic disc shape"""
    res = cv2.resize(u8, (64, 64))
    h = hog(res, orientations=9, pixels_per_cell=(8,8),
            cells_per_block=(2,2), visualize=False)
    step = len(h) // 36
    return [np.mean(h[i:i+step]) for i in range(0, 36*step, step)][:36]

def gabor_features(img_f32: np.ndarray) -> list:
    """Gabor filter — captures oriented nerve fiber textures"""
    feats = []
    for freq in [0.1, 0.3, 0.5]:
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            real, _ = gabor(img_f32, frequency=freq, theta=theta)
            feats += [real.mean(), real.std()]
    return feats

def get_core_features(img_f32: np.ndarray, include_gabor: bool = False) -> list:
    """Extracts the base set of features for any given image patch/layer."""
    # OPTIMIZATION: Compute uint8 representation ONCE per patch 
    # instead of doing it separately inside glcm, entropy, lbp, and hog!
    u8 = (img_f32 * 255).astype(np.uint8)
    
    feats = glcm_features(u8) + entropy_features(u8) + lbp_features(u8) + hog_features(u8)
    if include_gabor:
        feats += gabor_features(img_f32)
    return feats

def extract_one_image(processed_img: np.ndarray) -> list:
    """Full feature vector extraction for ONE image"""
    all_feats = []

    # IEMD decomposition into 3 IMFs + residue
    img_f = processed_img.astype(np.float32) / 255.0
    residue = img_f.copy()
    imfs = []
    
    for i in range(3):
        smooth = cv2.GaussianBlur(residue, (15,15), 2*(i+1))
        imfs.append(residue - smooth)
        residue = smooth
    imfs.append(residue)

    # Features from each IMF
    for imf in imfs:
        n = np.clip(imf, 0, None)
        if (mx := n.max()) > 0:
            n = n / mx
        all_feats += get_core_features(n)

    # Features from full preprocessed image
    norm = processed_img.astype(np.float32) / 255.0
    all_feats += get_core_features(norm, include_gabor=True)

    # Features from optic disc crop
    disc = crop_optic_disc(processed_img).astype(np.float32) / 255.0
    all_feats += get_core_features(disc, include_gabor=True)

    return all_feats
