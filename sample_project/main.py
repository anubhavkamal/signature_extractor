#!/usr/bin/python
"""Python main file."""
# -*- coding: utf-8 -*-
# -----------------------------------------
# author      : Ahmet Ozlu
# mail        : ahmetozlu93@gmail.com
# date        : 05.05.2019
# -----------------------------------------

import os
# pdf2image removed; using pypdfium2/PyMuPDF backends
import cv2
from PIL import Image

import color_correlation
import dewapper
import signature_extractor
import unsharpen


def convert_pdf_to_images(pdf_path, dpi=300):
    """Convert PDF to page images and return list of filenames, without Poppler.

    Order: prefer PyMuPDF (handles many transparency cases) then pypdfium2.
    """
    # First: PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        doc = fitz.open(pdf_path)
        image_files = []
        for i, page in enumerate(doc, start=1):
            # Preserve alpha to avoid transparency flattening artifacts, then composite on white
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=True, annots=False)
            img_rgba = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
            img_rgb = Image.alpha_composite(
                Image.new("RGBA", img_rgba.size, (255, 255, 255, 255)),
                img_rgba
            ).convert("RGB")
            out = f"page_{i}.jpg"
            img_rgb.save(out, "JPEG", quality=95, subsampling=0)
            image_files.append(out)
        doc.close()
        return image_files
    except ImportError:
        pass

    # Fallback: pypdfium2
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        n_pages = len(pdf)
        image_files = []
        scale = dpi / 72.0  # PDF points -> pixels
        for i in range(n_pages):
            page = pdf[i]
            pil_image = page.render(scale=scale, annotations=False).to_pil()
            # Ensure opaque background in case of transparency to avoid black blocks
            if pil_image.mode == "RGBA":
                pil_image = Image.alpha_composite(
                    Image.new("RGBA", pil_image.size, (255, 255, 255, 255)),
                    pil_image
                ).convert("RGB")
            else:
                pil_image = pil_image.convert("RGB")
            out = f"page_{i + 1}.jpg"
            pil_image.save(out, "JPEG", quality=95, subsampling=0)
            image_files.append(out)
        return image_files
    except ImportError as e:
        raise RuntimeError("No PDF rendering backend available. Install pymupdf or pypdfium2.") from e


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
