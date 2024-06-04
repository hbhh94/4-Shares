import cv2
import numpy as np

def process_image(I):
    # Check if the image is grayscale or color
    channels = 1 if len(I.shape) == 2 else I.shape[2]

    if channels == 1:
        I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Resize the image to fit into a 512x512 area
    I = cv2.resize(I, (516, 516))
    
    P = np.zeros((516, 516, 3), dtype=np.float64)
    P[:516, :516, :] = I

    block = 6

    s1 = np.zeros((516, 258, 3), dtype=np.float64)
    s2 = np.zeros((516, 258, 3), dtype=np.float64)
    s3 = np.zeros((516, 258, 3), dtype=np.float64)
    s4 = np.zeros((516, 258, 3), dtype=np.float64)

    for m in range(86):
        for n in range(86):
            x = (m * block)
            y = (n * block // 2)
            for c in range(3):
                bk = P[x:x+block, y*2:y*2+block, c]
                
                w1 = bk[:, 0].flatten()
                w2 = bk[:, 1].flatten()
                w3 = bk[:, 2].flatten()
                w4 = bk[:, 3].flatten()
                w5 = bk[:, 4].flatten()
                w6 = bk[:, 5].flatten()

                #deppending on this Matrix: [0 1 0 0 1 1
                #                            0 0 1 1 0 1
                #                            1 0 0 1 1 0
                #                            1 1 1 0 0 0]

                ww1 = np.vstack((w2, w5, w6)).T
                s1[x:x+block, y:y+block//2, c] = ww1

                ww2 = np.vstack((w3, w4, w6)).T
                s2[x:x+block, y:y+block//2, c] = ww2

                ww3 = np.vstack((w1, w4, w5)).T
                s3[x:x+block, y:y+block//2, c] = ww3

                ww4 = np.vstack((w1, w2, w3)).T
                s4[x:x+block, y:y+block//2, c] = ww4

    if channels == 1:
        s1 = cv2.cvtColor(s1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        s2 = cv2.cvtColor(s2.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        s3 = cv2.cvtColor(s3.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        s4 = cv2.cvtColor(s4.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    return s1, s2, s3, s4, channels


def encrypt_share(share, key):
    # Flatten the share to apply encryption
    flattened_share = share.flatten()
    
    # Repeat the key to match the length of the flattened share
    repeated_key = np.tile(key.flatten(), len(flattened_share) // len(key.flatten()) + 1)
    
    # Take only the required length of the repeated key
    repeated_key = repeated_key[:len(flattened_share)]
    
    # Encrypt the share using XOR operation with the repeated key
    encrypted_share = flattened_share ^ repeated_key
    
    # Reshape the encrypted share to its original shape
    encrypted_share = encrypted_share.reshape(share.shape)
    
    return encrypted_share


def reconstruct_image(s1, s2, s3, channels):
    if channels == 1:
        reconstructed_image = np.zeros((516, 516), dtype=np.float64)
    else:
        reconstructed_image = np.zeros((516, 516, channels), dtype=np.float64)
    
    block = 6

    for m in range(86):
        for n in range(86):
            x = m * block
            y = n * block // 2

            for c in range(channels):
                if channels == 1:
                    ww1 = s1[x:x+block, y:y+block//2]
                    ww2 = s2[x:x+block, y:y+block//2]
                    ww3 = s3[x:x+block, y:y+block//2]
                else:
                    ww1 = s1[x:x+block, y:y+block//2, c]
                    ww2 = s2[x:x+block, y:y+block//2, c]
                    ww3 = s3[x:x+block, y:y+block//2, c]

                w1 = ww3[:, 0]
                w2 = ww1[:, 0]
                w3 = ww2[:, 0]
                w4 = ww2[:, 1]
                w5 = ww3[:, 1]
                
                w6 = ww1[:, 1]

                bk = np.zeros((block, block))

                bk[:, 0] = w1
                bk[:, 1] = w2
                bk[:, 2] = w3
                bk[:, 3] = w4
                bk[:, 4] = w5
                bk[:, 5] = w6

                if channels == 1:
                    reconstructed_image[x:x+block, y*2:y*2+block] = bk
                else:
                    reconstructed_image[x:x+block, y*2:y*2+block, c] = bk

    # Convert reconstructed image to uint8 and crop to original size
    reconstructed_image = reconstructed_image[:512, :512].astype(np.uint8)
    return reconstructed_image


# Read the input image
image_path = 'lena.bmp'
I = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check if the image was successfully loaded
if I is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found or unable to read.")

I = I.astype(float)

s1, s2, s3, s4, channels = process_image(I)

shares = [s1, s2, s3, s4]

# Convert shares to uint8 before saving and save shares as images
for i, share in enumerate(shares):
    share = share.astype(np.uint8)
    cv2.imwrite(f's{i+1}.bmp', share)

# Load the shares
loaded_shares = []
for i in range(4):
    loaded_share = cv2.imread(f's{i+1}.bmp', cv2.IMREAD_UNCHANGED).astype(float)
    loaded_shares.append(loaded_share)

# Reconstruct the original image using any three of the four shares
combinations = [
    (loaded_shares[0], loaded_shares[1], loaded_shares[2]),
    (loaded_shares[0], loaded_shares[1], loaded_shares[3]),
    (loaded_shares[0], loaded_shares[2], loaded_shares[3]),
    (loaded_shares[1], loaded_shares[2], loaded_shares[3])
]

for i, (s1, s2, s3) in enumerate(combinations):
    reconstructed_image = reconstruct_image(s1, s2, s3, channels)
    cv2.imwrite(f'reconstructed_image_{i+1}.bmp', reconstructed_image)

# Encrypt shares using the unique features of the original image
key = np.random.randint(0, 256, size=I.shape, dtype=np.uint8)  # Generate a key based on the original image

for i, share in enumerate(loaded_shares):
    encrypted_share = encrypt_share(share.astype(np.uint8), key)
    cv2.imwrite(f'encrypted_s{i+1}.bmp', encrypted_share)
