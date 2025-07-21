import streamlit as st
import tempfile
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter

# Load YOLOv8 model
model = YOLO('weights/best.pt')  # Update this path

st.title("HPE CPP Project")
st.subheader("_Animal Species Classification in Forest through Camera Traps_")
st.markdown("Upload an image or video to detect species.")

# Helper to count species
def get_species_counts(result, model):
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[c] for c in class_ids]
    counts = Counter(class_names)
    return counts

# Helper to resize with letterbox padding
def letterbox(im, new_shape=(960, 960), color=(114, 114, 114)):
    shape = im.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2
    dh /= 2

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded

# Choose mode
mode = st.radio("Choose input type:", ('Image', 'Video'))

# IMAGE MODE
if mode == 'Image':
    image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if image_file:
        image = Image.open(image_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Resize image to 960x960 with padding
        img_array = np.array(image)
        resized_img = letterbox(img_array, new_shape=(960, 960))

        # Inference
        results = model(resized_img)
        annotated = results[0].plot()

        # Display output
        st.image(annotated, caption="Detected Image", use_column_width=True)

        # Species count
        counts = get_species_counts(results[0], model)
        if counts:
            st.subheader("📋 Detected Species:")
            for species, count in counts.items():
                st.markdown(f"- **{species}**: {count}")
        else:
            st.info("No animals detected.")

# VIDEO MODE
elif mode == 'Video':
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        st.video(video_file)

        output_path = "annotated_output.mp4"
        cap = cv2.VideoCapture(video_path)
        width = 960
        height = 960
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        species_total = {}
        stframe = st.empty()
        frame_count = 0

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                resized_frame = letterbox(frame, new_shape=(960, 960))
                results = model(resized_frame, verbose=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                # Count species in frame
                frame_counts = get_species_counts(results[0], model)
                for species, count in frame_counts.items():
                    species_total[species] = species_total.get(species, 0) + count

                frame_count += 1
                if frame_count % 10 == 0:
                    stframe.image(annotated_frame, caption=f"Frame {frame_count}", channels="BGR")

            cap.release()
            out.release()

        st.success("✅ Detection complete.")
        st.video(output_path)

        if species_total:
            st.subheader("📋 Detected Species Summary:")
            for species, count in species_total.items():
                st.markdown(f"- **{species}**: {count}")
        else:
            st.info("No animals detected in video.")
