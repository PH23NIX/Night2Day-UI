import streamlit as st
import tempfile
import os
import sys
import io
from PIL import Image
from modelUsageToCropping import Night2DayInference, CONFIG as CROPPING_CONFIG

# Logger to capture print output in UI
class StreamlitLogger(io.StringIO):
    def write(self, txt):
        super().write(txt)
        st.session_state["console_output"] += txt

# Initialize session state if not present
if "console_output" not in st.session_state:
    st.session_state["console_output"] = ""

if "cropping_results" not in st.session_state:
    st.session_state["cropping_results"] = None

# --------------- Sidebar: Navigation ----------------
page = st.sidebar.radio("Go to", ["Test Pipeline", "Model Usage to Cropping"])

# ==============================
# TEST PIPELINE UI PAGE
# ==============================
if page == "Test Pipeline":
    st.title("Seatbelt/Person Detection Test")

    from pipeline import SeatbeltDetectionPipeline, CONFIG as TEST_CONFIG

    st.sidebar.header("Configuration Settings")

    TEST_CONFIG["person_detection_model"] = st.sidebar.text_input(
        "Person Detection Model Path", value=TEST_CONFIG["person_detection_model"]
    )
    TEST_CONFIG["seatbelt_model_path"] = st.sidebar.text_input(
        "Seatbelt Model Path", value=TEST_CONFIG["seatbelt_model_path"]
    )
    TEST_CONFIG["person_confidence"] = st.sidebar.number_input(
        "Person Confidence", 0.0, 1.0,
        value=float(TEST_CONFIG["person_confidence"]), step=0.01
    )
    TEST_CONFIG["seatbelt_confidence"] = st.sidebar.number_input(
        "Seatbelt Confidence", 0.0, 1.0,
        value=float(TEST_CONFIG["seatbelt_confidence"]), step=0.01
    )
    TEST_CONFIG["iou_threshold"] = st.sidebar.number_input(
        "IoU Threshold", 0.0, 1.0,
        value=float(TEST_CONFIG["iou_threshold"]), step=0.01
    )
    TEST_CONFIG["iou_match_threshold"] = st.sidebar.number_input(
        "IoU Match Threshold", 0.0, 1.0,
        value=float(TEST_CONFIG["iou_match_threshold"]), step=0.01
    )
    TEST_CONFIG["save_visualization"] = st.sidebar.checkbox(
        "Save Visualizations", value=TEST_CONFIG.get("save_visualization", True)
    )
    st.sidebar.text(f"Device: {TEST_CONFIG['device']}")

    st.markdown("---")

    st.subheader("Upload Comparison Images & Labels")
    comp_files = st.file_uploader(
        "Comparison Images (.png/.jpg)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    label_files = st.file_uploader(
        "Label Files (.txt)", type="txt", accept_multiple_files=True
    )
    TEST_CONFIG["output_dir"] = st.text_input(
        "Output Directory (for evaluation results)",
        value=TEST_CONFIG["output_dir"]
    )
    st.markdown("---")

    if st.button("Run Test Pipeline"):
        st.session_state["console_output"] = ""
        logger = StreamlitLogger()
        sys.stdout = logger

        if not comp_files or not label_files:
            st.error("Please upload both comparison images and label files.")
        else:
            workspace = tempfile.mkdtemp()
            for f in comp_files:
                with open(os.path.join(workspace, f.name), "wb") as out:
                    out.write(f.getbuffer())
            for f in label_files:
                with open(os.path.join(workspace, f.name), "wb") as out:
                    out.write(f.getbuffer())
            TEST_CONFIG["comparison_images_dir"] = workspace
            TEST_CONFIG["labels_dir"] = workspace
            os.makedirs(TEST_CONFIG["output_dir"], exist_ok=True)

            try:
                pipeline = SeatbeltDetectionPipeline(TEST_CONFIG)
                report = pipeline.run()
                st.success("Evaluation complete!")
                st.write("### Detection Metrics")
                st.json(report["detection_metrics"])
                st.write("### mAP Metrics")
                st.json(report["map_metrics"])
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

        sys.stdout = sys.__stdout__

    st.subheader("Console Output")
    st.text_area("", value=st.session_state["console_output"], height=250)

# ========================================
# MODEL USAGE TO CROPPING UI PAGE
# ========================================
elif page == "Model Usage to Cropping":
    st.title("Model Usage to Cropping")

    st.sidebar.header("Cropping / Inference Configuration")

    CROPPING_CONFIG["yolo_model_path"] = st.sidebar.text_input(
        "YOLO Model Path", value=CROPPING_CONFIG["yolo_model_path"]
    )
    CROPPING_CONFIG["model_path"] = st.sidebar.text_input(
        "CycleGAN Model Path (.pth)", value=CROPPING_CONFIG["model_path"]
    )
    CROPPING_CONFIG["yolo_confidence"] = st.sidebar.number_input(
        "YOLO Confidence", 0.0, 1.0,
        value=float(CROPPING_CONFIG["yolo_confidence"]), step=0.01
    )
    CROPPING_CONFIG["yolo_iou_threshold"] = st.sidebar.number_input(
        "YOLO IoU Threshold", 0.0, 1.0,
        value=float(CROPPING_CONFIG["yolo_iou_threshold"]), step=0.01
    )
    CROPPING_CONFIG["yolo_min_area"] = st.sidebar.number_input(
        "Min Vehicle Area", value=int(CROPPING_CONFIG["yolo_min_area"])
    )

    CROPPING_CONFIG["discriminator_strictness"] = st.sidebar.number_input(
        "Discriminator Threshold", 0.0, 1.0,
        value=float(CROPPING_CONFIG["discriminator_strictness"]), step=0.01
    )

    CROPPING_CONFIG["img_size"] = st.sidebar.number_input(
        "Generator Image Size", value=int(CROPPING_CONFIG["img_size"])
    )

    CROPPING_CONFIG["batch_processing"] = st.sidebar.checkbox(
        "Batch Processing", value=bool(CROPPING_CONFIG["batch_processing"])
    )

    CROPPING_CONFIG["output_dir"] = st.text_input(
        "Output Directory for Cropping Results",
        value=CROPPING_CONFIG["output_dir"]
    )

    st.markdown("---")
    st.subheader("Upload Night Images for Cropping + Inference")

    night_imgs = st.file_uploader(
        "Upload Night Images (.png/.jpg)",
        type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if st.button("Run Cropping Inference on All Images"):
        st.session_state["console_output"] = ""
        logger = StreamlitLogger()
        sys.stdout = logger

        if not night_imgs:
            st.error("⚠ Please upload at least one image to run inference.")
        else:
            workspace = tempfile.mkdtemp()
            for f in night_imgs:
                with open(os.path.join(workspace, f.name), "wb") as out:
                    out.write(f.getbuffer())

            CROPPING_CONFIG["input_dir"] = workspace
            os.makedirs(CROPPING_CONFIG["output_dir"], exist_ok=True)

            try:
                inference = Night2DayInference(CROPPING_CONFIG)
                inference.run()
                st.session_state["cropping_results"] = CROPPING_CONFIG["output_dir"]
                st.success("Inference + Cropping completed!")
            except Exception as e:
                st.error(f"Cropping inference error: {e}")

        sys.stdout = sys.__stdout__

    st.subheader("Cropping Console Output")
    st.text_area("", value=st.session_state["console_output"], height=250)

    # === Show Results Gallery ===
    if st.session_state["cropping_results"]:
        outdir = st.session_state["cropping_results"]
        st.write("### 👍 Cropping Results Directory")
        st.write(f"`{outdir}`")

        # List full comparison outputs
        st.write("### 📸 Full Image Comparisons")
        full_cmp_dir = os.path.join(outdir, "full_image_comparisons")
        if os.path.exists(full_cmp_dir):
            for fn in sorted(os.listdir(full_cmp_dir)):
                path = os.path.join(full_cmp_dir, fn)
                st.image(path, caption=fn)
                st.download_button("⬇️ Download " + fn, open(path, "rb").read(), file_name=fn)
        else:
            st.write("No full comparisons generated yet.")

        # List crops
        st.write("###Vehicle Crops")
        crop_dir = os.path.join(outdir, "vehicle_crops")
        if os.path.exists(crop_dir):
            for fn in sorted(os.listdir(crop_dir)):
                path = os.path.join(crop_dir, fn)
                st.image(path, caption=fn)
                st.download_button("⬇️ Download " + fn, open(path, "rb").read(), file_name=fn)
        else:
            st.write("No vehicle crops found.")

        # List converted day images
        st.write("###Converted Day Images")
        conv_dir = os.path.join(outdir, "converted_images")
        if os.path.exists(conv_dir):
            for fn in sorted(os.listdir(conv_dir)):
                path = os.path.join(conv_dir, fn)
                st.image(path, caption=fn)
                st.download_button("⬇️ Download " + fn, open(path, "rb").read(), file_name=fn)
        else:
            st.write("No converted images found.")

st.write("---")

