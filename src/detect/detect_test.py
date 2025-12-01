import cv2


# cv2.imshow("Test Image", raw_image)
for i in range(42, 73):

    test_image_path = f"../../data/frame/frame_{i:04d}.jpg"
    
    raw_image = cv2.imread(test_image_path)

    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 63, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Test Image", binary_image)

    gravity_center = [0, 0]
    total_pixels = 0

    for i in range(binary_image.shape[1]):
        for j in range(binary_image.shape[0]):
            if binary_image[j, i] == 255:
                gravity_center[0] += i
                gravity_center[1] += j
                total_pixels += 1

    if total_pixels == 0:
        continue
    gravity_center[0] /= total_pixels
    gravity_center[1] /= total_pixels

    cv2.circle(raw_image, (int(gravity_center[0]), int(gravity_center[1])), 5, (0, 0, 255), -1)
    cv2.imshow("Test Image with Gravity Center", raw_image)
    cv2.waitKey(50)
    print("Gravity Center:", gravity_center)

cv2.waitKey(0)