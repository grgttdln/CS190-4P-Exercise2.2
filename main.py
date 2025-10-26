import cv2
import numpy as np
import os
import argparse

MIN_MATCH_COUNT = 10

def load_images_from_folder(folder):
    images = []
    files = sorted(os.listdir(folder))
    for filename in files:
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    return images

def detect_and_compute(img, method='sift'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'sift':
        detector = cv2.SIFT_create()
    elif method == 'akaze':
        detector = cv2.AKAZE_create()
    elif method == 'orb':
        detector = cv2.ORB_create()
    elif method == 'brisk':
        detector = cv2.BRISK_create()
    else:
        raise ValueError(f"Unknown method: {method}")
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(des1, des2, method='sift', use_knn=False):
    # SIFT / BRISK / AKAZE (float descriptors)
    if method in ['sift', 'brisk', 'akaze']:
        matcher = cv2.BFMatcher()
        if use_knn:
            matches = matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            return good
        else:
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches

    # ORB (binary descriptors)
    elif method == 'orb':
        # crossCheck=True = symmetric matching (more reliable)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    else:
        raise ValueError(f"Unknown method: {method}")

def stitch_images(img1, img2, custom_method=None):
    method = 'sift'
    if custom_method == 'akaze':
        method = 'akaze'
    elif custom_method == 'orb':
        method = 'orb'
    elif custom_method == 'brisk':
        method = 'brisk'

    kp1, des1 = detect_and_compute(img1, method)
    kp2, des2 = detect_and_compute(img2, method)
    matches = match_features(des1, des2, method, use_knn=True)

    if len(matches) < MIN_MATCH_COUNT:
        print(f"Not enough matches ({len(matches)}) to stitch images.")
        return None, None

    # Draw matches for visualization
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if custom_method == "rho":
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    elif custom_method == "lmeds":
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
    elif custom_method == "blend":
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    elif custom_method == "filter":
        strong_matches = [m for m in matches if m.distance < 0.6 * np.median([x.distance for x in matches])]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in strong_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in strong_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("Homography could not be computed.")
        return None, match_img

    # Warp and combine images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    dst_corners = cv2.perspectiveTransform(pts_img1, M)

    all_corners = np.concatenate((dst_corners, np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
    translation = [-xmin, -ymin]
    H_translation = np.array([[1,0,translation[0]], [0,1,translation[1]], [0,0,1]])

    panorama = cv2.warpPerspective(img1, H_translation.dot(M), (xmax-xmin, ymax-ymin))
    panorama[translation[1]:h2+translation[1], translation[0]:w2+translation[0]] = img2

    if custom_method == "blend":
        alpha = 0.5
        blend_area = panorama[translation[1]:h2+translation[1], translation[0]:w2+translation[0]]
        panorama[translation[1]:h2+translation[1], translation[0]:w2+translation[0]] = cv2.addWeighted(blend_area, alpha, img2, 1-alpha, 0)

    return panorama, match_img

def main():
    parser = argparse.ArgumentParser(description="Panorama Stitching with Custom Methods")
    parser.add_argument('--dir', required=True, help="Folder containing images")
    parser.add_argument('--custom', default=None, help="Custom algorithm: rho, lmeds, blend, filter, akaze, orb, brisk")
    parser.add_argument('--output', default='panorama.png', help="Output panorama filename")
    args = parser.parse_args()

    images = load_images_from_folder(args.dir)
    if len(images) < 2:
        print("Need at least two images to stitch.")
        return

    panorama = images[0]
    all_match_imgs = []

    for i in range(1, len(images)):
        panorama, match_img = stitch_images(panorama, images[i], custom_method=args.custom)
        if panorama is None:
            print("Stitching failed.")
            return
        if match_img is not None:
            all_match_imgs.append(match_img)

    # Resize all match images to the same width for vertical stacking
    if all_match_imgs:
        max_width = max(img.shape[1] for img in all_match_imgs)
        resized_imgs = []
        for img in all_match_imgs:
            h, w = img.shape[:2]
            scale = max_width / w
            new_h = int(h * scale)
            resized = cv2.resize(img, (max_width, new_h))
            resized_imgs.append(resized)

        combined_matches = np.vstack(resized_imgs)
        cv2.imshow("All Feature Matches", combined_matches)
        cv2.waitKey(0)
        cv2.destroyWindow("All Feature Matches")

    cv2.imwrite(args.output, panorama)
    print(f"Panorama saved to {args.output}")

if __name__ == "__main__":
    main()
