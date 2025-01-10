import cv2
import easyocr
import re
import os
import matplotlib.pyplot as plt

LICENSE_PLATE_PATTERNS = {
    "USA": r"[A-Z0-9]{1,7}",
    "India": r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}",
    "Nepal": r"[A-Z]{2}-[0-9]{1,4}-[A-Z]{1,2}",
    "Australia": r"[A-Z]{1,3}[0-9]{1,4}",
    "Canada": r"[A-Z0-9]{1,7}"
}

def detect_country(text):
    for country, pattern in LICENSE_PLATE_PATTERNS.items():
        if re.match(pattern, text.replace(" ", "")):
            return country
    return "Unknown"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray, image

def process_frame(frame, reader):
    gray, frame = preprocess_image(frame)
    edges = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 1.5 <= aspect_ratio <= 5.0:
                roi = frame[y:y + h, x:x + w]
                if roi.shape[0] < 20 or roi.shape[1] < 80:
                    continue
                result = reader.readtext(roi)
                if result:
                    for (bbox, text, prob) in result:
                        if prob < 0.2:
                            continue
                        text = ''.join(e for e in text if e.isalnum() or e.isspace())
                        if len(text) >= 4:
                            country = detect_country(text)
                            detected_plates.append((text, country, prob))
                            plate_info = f"{text} ({country}) {prob:.2f}"
                            cv2.putText(frame, plate_info, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, detected_plates

def display_frame_with_matplotlib(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()

def process_image(image_path, reader):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return
    processed_frame, plates = process_frame(image, reader)
    if plates:
        for plate, country, prob in plates:
            print(f"Detected Plate: {plate} - Country: {country} - Confidence: {prob:.2f}")
    display_frame_with_matplotlib(processed_frame)

def play_video(video_path, reader, use_matplotlib=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        processed_frame, plates = process_frame(frame, reader)
        if plates:
            for plate, country, prob in plates:
                print(f"Detected Plate: {plate} - Country: {country} - Confidence: {prob:.2f}")
        if use_matplotlib:
            display_frame_with_matplotlib(processed_frame)
        else:
            cv2.imshow("License Plate Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    if not use_matplotlib:
        cv2.destroyAllWindows()

def main():
    print("License Plate Recognition System")
    print("===============================")
    print("1. Process Video")
    print("2. Use Webcam")
    print("3. Process Image")
    choice = input("Enter your choice (1-3): ").strip()
    reader = easyocr.Reader(['en'])
    try:
        if choice == '1':
            video_path = input("Enter video file path: ").strip()
            try:
                play_video(video_path, reader)
            except cv2.error:
                print("Falling back to Matplotlib for video display...")
                play_video(video_path, reader, use_matplotlib=True)
        elif choice == '2':
            try:
                play_video(0, reader)
            except cv2.error:
                print("Falling back to Matplotlib for webcam feed...")
                play_video(0, reader, use_matplotlib=True)
        elif choice == '3':
            image_path = input("Enter image file path: ").strip()
            process_image(image_path, reader)
    except KeyboardInterrupt:
        print("Those magical keys interrupted me")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
