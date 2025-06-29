"""
PDF preprocessing module for converting PDFs to images.
Handles PDF ingestion, slide conversion, and image resizing with aspect ratio preservation.
"""
import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import time

from pdf2image import convert_from_path
from PIL import Image, ImageOps
import numpy as np

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF to image conversion with proper resizing."""
    
    def __init__(self, output_dir: str = "processed_images", target_size: Tuple[int, int] = (1500, 1500)):
        """
        Initialize PDF processor.
        
        Args:
            output_dir: Directory to save processed images
            target_size: Target size for output images (width, height)
        """
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[str]:
        """
        Convert PDF to images.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for image conversion (higher = better quality)
            
        Returns:
            List of image file paths
        """
        try:
            pdf_name = Path(pdf_path).stem
            logger.info(f"Converting PDF {pdf_name} to images...")
            
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=dpi)
            image_paths = []
            
            for i, page in enumerate(pages):
                # Process and resize image
                processed_image = self._resize_with_padding(page)
                
                # Save image
                image_filename = f"{pdf_name}_slide_{i+1:03d}.png"
                image_path = self.output_dir / image_filename
                processed_image.save(image_path, "PNG", quality=95)
                image_paths.append(str(image_path))
                
                logger.debug(f"Saved slide {i+1} as {image_filename}")
            
            logger.info(f"Successfully converted {len(pages)} slides from {pdf_name}")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {str(e)}")
            raise
    
    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """
        Resize image to target size while preserving aspect ratio.
        Adds padding as needed to maintain the target dimensions.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Resized image with padding
        """
        # Calculate scaling factor to fit within target size
        width_ratio = self.target_size[0] / image.width
        height_ratio = self.target_size[1] / image.height
        scale_factor = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and white background
        final_image = Image.new('RGB', self.target_size, (255, 255, 255))
        
        # Calculate position to center the resized image
        x_offset = (self.target_size[0] - new_width) // 2
        y_offset = (self.target_size[1] - new_height) // 2
        
        # Paste resized image onto the centered position
        final_image.paste(resized_image, (x_offset, y_offset))
        
        return final_image
    
    def process_pdf_batch(self, pdf_paths: List[str], dpi: int = 200) -> dict:
        """
        Process multiple PDFs in batch.
        
        Args:
            pdf_paths: List of PDF file paths
            dpi: DPI for image conversion
            
        Returns:
            Dictionary mapping PDF names to their image paths
        """
        results = {}
        total_start_time = time.time()
        
        for pdf_path in pdf_paths:
            try:
                start_time = time.time()
                image_paths = self.convert_pdf_to_images(pdf_path, dpi)
                processing_time = time.time() - start_time
                
                pdf_name = Path(pdf_path).stem
                results[pdf_name] = {
                    'image_paths': image_paths,
                    'slide_count': len(image_paths),
                    'processing_time': processing_time,
                    'success': True,
                    'error': None
                }
                
                logger.info(f"Processed {pdf_name}: {len(image_paths)} slides in {processing_time:.2f}s")
                
            except Exception as e:
                pdf_name = Path(pdf_path).stem
                results[pdf_name] = {
                    'image_paths': [],
                    'slide_count': 0,
                    'processing_time': 0,
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"Failed to process {pdf_name}: {str(e)}")
        
        total_time = time.time() - total_start_time
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        
        return results
    
    def get_image_info(self, image_path: str) -> dict:
        """
        Get information about a processed image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_bytes': os.path.getsize(image_path)
                }
        except Exception as e:
            logger.error(f"Error getting image info for {image_path}: {str(e)}")
            return {}


def setup_pdf_processor(output_dir: str = "processed_images") -> PDFProcessor:
    """
    Factory function to create and configure PDF processor.
    
    Args:
        output_dir: Directory for processed images
        
    Returns:
        Configured PDFProcessor instance
    """
    processor = PDFProcessor(output_dir=output_dir)
    logger.info(f"PDF processor initialized with output directory: {output_dir}")
    return processor


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    processor = setup_pdf_processor()
    
    # Example: process a single PDF
    # image_paths = processor.convert_pdf_to_images("example.pdf")
    # print(f"Generated {len(image_paths)} images")
