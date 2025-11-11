#!/usr/bin/python
"""Python main file."""
# -*- coding: utf-8 -*-
# -----------------------------------------
# author      : Ahmet Ozlu
# mail        : ahmetozlu93@gmail.com
# date        : 05.05.2019
# -----------------------------------------

import os
from pdf2image import convert_from_path
import cv2

import color_correlation
import dewapper
import signature_extractor
import unsharpen


def convert_pdf_to_images(pdf_path):
    """Convert PDF to page images and return list of filenames."""
    images = convert_from_path(pdf_path)
    image_files = []
    for i, image in enumerate(images):
        image_file = f"page_{i + 1}.jpg"
        image.save(image_file, "JPEG")
        image_files.append(image_file)
    return image_files


# Resolve base dir and locate the PDF robustly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def resolve_pdf_path(filename):
    candidates = [
        os.path.join(BASE_DIR, filename),                 # e.g. .../signature_extractor/signature_extractor
        os.path.join(os.path.dirname(BASE_DIR), filename),# e.g. project root .../signature_extractor
        os.path.join(os.getcwd(), filename),              # current working directory
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find '{filename}' in any of: {candidates}")

PDF_FILENAME = "Job Offer-Applied AI ML Associate.pdf"
pdf_path = resolve_pdf_path(PDF_FILENAME)

# Convert PDF into images
try:
    image_files = convert_pdf_to_images(pdf_path)
except Exception as e:
    print("type error: " + str(e))
    print(f"ERROR CONVERTING PDF! Tried at: {pdf_path}")
    image_files = []

if not image_files:
    # Nothing to process
    raise SystemExit(1)

for idx, image_file in enumerate(image_files):
    source_image = cv2.imread(image_file)
    if source_image is None:
        print(f"type error: could not read generated image file: {image_file}")
        print("ERROR READING PAGE IMAGE!")
        continue

    img = None

    # Step 1: Dewarp/crop
    try:
        img = dewapper.dewarp_book(source_image)
        cv2.imwrite(f"step 1 - page_{idx + 1}_dewarped.jpg", img)
        print(f"- step1 (cropping with the margins + book dewarping) for page {idx + 1}: OK")
    except Exception as e:
        print("type error: " + str(e))
        print("FALLBACK: using original page without dewarping.")
        img = source_image.copy()
        # persist step 1 output even when dewarp fails
        try:
            cv2.imwrite(f"step 1 - page_{idx + 1}_dewarped.jpg", img)
        except Exception as ee:
            print("type error: " + str(ee))
        # proceed with downstream steps on fallback image

    # Step 2: Signature extraction
    try:
        if img is None:
            raise ValueError("No image produced from dewarping step")
        # Ensure grayscale safely before extraction
        if len(img.shape) == 3:
            # 3 or 4 channel images
            if img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                raise ValueError(f"Unexpected channel count: {img.shape[2]}")
        elif len(img.shape) == 2:
            gray = img
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        img = signature_extractor.extract_signature(gray)
        cv2.imwrite(f"step 2 - page_{idx + 1}_signature_extracted.jpg", img)
        print(f"- step2 (signature extractor) for page {idx + 1}: OK")
    except Exception as e:
        print("type error: " + str(e))
        print("ERROR IN SIGNATURE EXTRACTION! PLEASE CHECK LIGHTNING, SHADOW, ZOOM LEVEL AND ETC. OF YOUR INPUT BOOK IMAGE!")
        # Skip remaining steps for this page if signature extraction failed
        continue

    # Step 3: Unsharpen mask
    try:
        img = unsharpen.unsharpen_mask(img)
        cv2.imwrite(f"step 3 - page_{idx + 1}_unsharpen_mask.jpg", img)
        print(f"- step3 (unsharpening mask) for page {idx + 1}: OK")
    except Exception as e:
        print("type error: " + str(e))
        print("ERROR IN BOOK UNSHARPING MASK! PLEASE CHECK LIGHTNING, SHADOW, ZOOM LEVEL AND ETC. OF YOUR INPUT BOOK IMAGE!")
        continue

    # Step 4: Color correlation
    try:
        img = color_correlation.funcBrightContrast(img)
        cv2.imwrite(f"step 4 - page_{idx + 1}_color_correlated.jpg", img)
        print(f"- step4 (color correlation) for page {idx + 1}: OK")
    except Exception as e:
        print("type error: " + str(e))
        print("ERROR IN BOOK COLOR CORRELATION! PLEASE CHECK LIGHTNING, SHADOW, ZOOM LEVEL AND ETC. OF YOUR INPUT BOOK IMAGE!")
        continue

    # Final output for this page
    try:
        cv2.imwrite(f"output_page_{idx + 1}.jpg", img)
    except Exception as e:
        print("type error: " + str(e))
        print(f"ERROR SAVING FINAL OUTPUT FOR PAGE {idx + 1}!")
