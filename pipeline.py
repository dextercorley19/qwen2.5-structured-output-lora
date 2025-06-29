"""
Main pipeline orchestrator for pitch deck analysis and VLM fine-tuning.
Coordinates all components: PDF processing, Gemini analysis, tracking, and dataset preparation.
"""
import os
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from dotenv import load_dotenv

from pdf_processor import PDFProcessor, setup_pdf_processor
from gemini_analyzer import GeminiAnalyzer, setup_gemini_analyzer
from langfuse_tracker import LangfuseTracker, setup_langfuse_tracker
from dataset_preparer import DatasetPreparer
from schemas import PitchDeckAnalysis, ProcessingResult

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PitchDeckPipeline:
    """Main pipeline orchestrator for processing pitch decks and preparing VLM training data."""
    
    def __init__(self, 
                 pdf_dir: str = "PitchDecks",
                 output_dir: str = "processed_images",
                 training_data_dir: str = "training_data"):
        """
        Initialize the pipeline.
        
        Args:
            pdf_dir: Directory containing PDF files
            pdf_dir: Directory for processed images
            training_data_dir: Directory for training data
        """
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.training_data_dir = Path(training_data_dir)
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.training_data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pdf_processor = None
        self.gemini_analyzer = None
        self.langfuse_tracker = None
        self.dataset_preparer = None
        
        # Pipeline state
        self.session_id = None
        self.processing_results = {}
        self.failed_pdfs = []
        
        logger.info(f"Pipeline initialized with PDF directory: {self.pdf_dir}")
    
    def setup_components(self):
        """Setup all pipeline components."""
        try:
            logger.info("Setting up pipeline components...")
            
            # Setup PDF processor
            self.pdf_processor = setup_pdf_processor(str(self.output_dir))
            
            # Setup Gemini analyzer
            self.gemini_analyzer = setup_gemini_analyzer()
            
            # Setup Langfuse tracker (optional)
            try:
                self.langfuse_tracker = setup_langfuse_tracker()
            except Exception as e:
                logger.warning(f"Langfuse setup failed, continuing without tracking: {e}")
                self.langfuse_tracker = None
            
            # Setup dataset preparer
            self.dataset_preparer = DatasetPreparer(str(self.training_data_dir))
            
            logger.info("All components setup successfully!")
            
        except Exception as e:
            logger.error(f"Failed to setup components: {str(e)}")
            raise
    
    def get_pdf_files(self) -> List[str]:
        """Get list of PDF files to process."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        return [str(pdf_file) for pdf_file in pdf_files]
    
    def start_session(self, session_name: Optional[str] = None) -> str:
        """
        Start a new Langfuse session for tracking.
        
        Args:
            session_name: Name for the session
            
        Returns:
            Session ID
        """
        if not session_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"pitch_deck_pipeline_{timestamp}"
        
        pdf_files = self.get_pdf_files()
        
        session_metadata = {
            "pipeline_version": "1.0",
            "total_pdfs": len(pdf_files),
            "pdf_directory": str(self.pdf_dir),
            "output_directory": str(self.output_dir),
            "training_data_directory": str(self.training_data_dir),
            "start_time": datetime.now().isoformat()
        }
        
        self.session_id = self.langfuse_tracker.create_session(session_name, session_metadata) if self.langfuse_tracker else f"session_{int(time.time())}"
        logger.info(f"Started session: {session_name} (ID: {self.session_id})")
        
        return self.session_id
    
    def process_single_pdf(self, pdf_path: str) -> ProcessingResult:
        """
        Process a single PDF file through the entire pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessingResult object
        """
        # Setup components if not already done
        if self.pdf_processor is None:
            self.setup_components()
            
        pdf_name = Path(pdf_path).stem
        start_time = time.time()
        
        logger.info(f"Processing PDF: {pdf_name}")
        
        try:
            # Step 1: Convert PDF to images
            logger.info(f"Step 1: Converting {pdf_name} to images...")
            image_paths = self.pdf_processor.convert_pdf_to_images(pdf_path)
            
            # Track PDF processing
            pdf_result = {
                "image_paths": image_paths,
                "slide_count": len(image_paths),
                "processing_time": time.time() - start_time,
                "success": True,
                "error": None
            }
            
            if self.langfuse_tracker:
                self.langfuse_tracker.track_pdf_processing(pdf_name, self.session_id, pdf_result)
            
            # Step 2: Analyze with Gemini
            logger.info(f"Step 2: Analyzing {pdf_name} with Gemini...")
            gemini_start_time = time.time()
            
            analysis = self.gemini_analyzer.analyze_pitch_deck(image_paths, pdf_name)
            gemini_processing_time = time.time() - gemini_start_time
            
            # Track Gemini analysis
            if self.langfuse_tracker:
                self.langfuse_tracker.track_gemini_analysis(
                    pdf_name, self.session_id, image_paths, analysis, 
                    gemini_processing_time, None if analysis else "Analysis failed"
                )
            
            if not analysis:
                raise ValueError("Gemini analysis failed")
            
            # Step 3: Create training entries
            logger.info(f"Step 3: Creating training entries for {pdf_name}...")
            training_entries = self.dataset_preparer.create_training_entries(
                image_paths, analysis, pdf_name
            )
            
            # Create processing result
            total_time = time.time() - start_time
            result = ProcessingResult(
                pdf_filename=pdf_name,
                total_slides=len(image_paths),
                images_generated=image_paths,
                gemini_analysis=analysis,
                processing_time=total_time,
                success=True,
                error_message=None
            )
            
            logger.info(f"Successfully processed {pdf_name}: {len(image_paths)} slides, "
                       f"{len(training_entries)} training entries in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Failed to process {pdf_name}: {str(e)}"
            logger.error(error_msg)
            
            # Track the failure
            pdf_result = {
                "image_paths": [],
                "slide_count": 0,
                "processing_time": total_time,
                "success": False,
                "error": str(e)
            }
            
            if self.langfuse_tracker:
                self.langfuse_tracker.track_pdf_processing(pdf_name, self.session_id, pdf_result)
            
            result = ProcessingResult(
                pdf_filename=pdf_name,
                total_slides=0,
                images_generated=[],
                gemini_analysis=None,
                processing_time=total_time,
                success=False,
                error_message=str(e)
            )
            
            return result
    
    def run_full_pipeline(self, max_pdfs: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline on all PDF files.
        
        Args:
            max_pdfs: Maximum number of PDFs to process (for testing)
            
        Returns:
            Summary of pipeline execution
        """
        if not self.session_id:
            self.start_session()
        
        pdf_files = self.get_pdf_files()
        if max_pdfs:
            pdf_files = pdf_files[:max_pdfs]
            logger.info(f"Limited processing to first {max_pdfs} PDFs")
        
        pipeline_start_time = time.time()
        successful_pdfs = 0
        total_slides = 0
        total_training_entries = 0
        
        logger.info(f"Starting pipeline execution for {len(pdf_files)} PDFs...")
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Processing PDF {i}/{len(pdf_files)}: {Path(pdf_path).name}")
            
            try:
                result = self.process_single_pdf(pdf_path)
                self.processing_results[result.pdf_filename] = result
                
                if result.success:
                    successful_pdfs += 1
                    total_slides += result.total_slides
                else:
                    self.failed_pdfs.append(result.pdf_filename)
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {pdf_path}: {str(e)}")
                self.failed_pdfs.append(Path(pdf_path).stem)
        
        # Get dataset statistics
        dataset_stats = self.dataset_preparer.get_dataset_statistics()
        total_training_entries = dataset_stats.get("total_entries", 0)
        
        # Save dataset
        logger.info("Saving training dataset...")
        dataset_json_path = self.dataset_preparer.save_dataset("pitch_deck_dataset.json")
        
        # Create Hugging Face datasets
        if total_training_entries > 0:
            try:
                train_dataset, val_dataset = self.dataset_preparer.create_huggingface_dataset()
                hf_dataset_path = self.dataset_preparer.save_huggingface_dataset(
                    train_dataset, val_dataset, "pitch_deck_dataset"
                )
                logger.info(f"Saved Hugging Face dataset to {hf_dataset_path}")
            except Exception as e:
                logger.error(f"Failed to create Hugging Face dataset: {str(e)}")
                hf_dataset_path = None
        else:
            hf_dataset_path = None
        
        # Calculate final statistics
        total_time = time.time() - pipeline_start_time
        success_rate = successful_pdfs / len(pdf_files) if pdf_files else 0
        
        # Pipeline summary
        summary = {
            "total_pdfs": len(pdf_files),
            "successful_pdfs": successful_pdfs,
            "failed_pdfs": len(self.failed_pdfs),
            "success_rate": success_rate,
            "total_slides": total_slides,
            "total_training_entries": total_training_entries,
            "total_processing_time": total_time,
            "dataset_json_path": dataset_json_path,
            "hf_dataset_path": hf_dataset_path,
            "failed_pdf_list": self.failed_pdfs,
            "dataset_statistics": dataset_stats,
            "start_time": pipeline_start_time,
            "end_time": time.time()
        }
        
        # Track pipeline summary
        if self.langfuse_tracker:
            self.langfuse_tracker.track_pipeline_summary(
                self.session_id, len(pdf_files), successful_pdfs, total_time, summary
            )
        
        # Track dataset creation
        if self.langfuse_tracker:
            self.langfuse_tracker.track_dataset_creation(
                self.session_id, total_training_entries, total_training_entries, 
                dataset_json_path
            )
        
        # Save summary to file
        summary_path = self.training_data_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Flush all tracking data
        if self.langfuse_tracker:
            self.langfuse_tracker.flush()
        
        logger.info(f"Pipeline execution completed!")
        logger.info(f"Success rate: {success_rate:.2%} ({successful_pdfs}/{len(pdf_files)} PDFs)")
        logger.info(f"Total slides processed: {total_slides}")
        logger.info(f"Total training entries: {total_training_entries}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Summary saved to: {summary_path}")
        
        return summary
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            "session_id": self.session_id,
            "total_processed": len(self.processing_results),
            "successful": len([r for r in self.processing_results.values() if r.success]),
            "failed": len(self.failed_pdfs),
            "failed_pdfs": self.failed_pdfs,
            "dataset_entries": self.dataset_preparer.get_dataset_statistics() if self.dataset_preparer else {}
        }


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pitch Deck Analysis and VLM Fine-tuning Pipeline")
    parser.add_argument("--pdf-dir", default="PitchDecks", help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="processed_images", help="Directory for processed images")
    parser.add_argument("--training-dir", default="training_data", help="Directory for training data")
    parser.add_argument("--max-pdfs", type=int, help="Maximum number of PDFs to process")
    parser.add_argument("--session-name", help="Custom session name for tracking")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = PitchDeckPipeline(
            pdf_dir=args.pdf_dir,
            output_dir=args.output_dir,
            training_data_dir=args.training_dir
        )
        
        # Setup components
        pipeline.setup_components()
        
        # Start session
        pipeline.start_session(args.session_name)
        
        # Run pipeline
        summary = pipeline.run_full_pipeline(max_pdfs=args.max_pdfs)
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"Total PDFs: {summary['total_pdfs']}")
        print(f"Successful: {summary['successful_pdfs']}")
        print(f"Failed: {summary['failed_pdfs']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Total Slides: {summary['total_slides']}")
        print(f"Training Entries: {summary['total_training_entries']}")
        print(f"Processing Time: {summary['total_processing_time']:.2f}s")
        print(f"Dataset Path: {summary['dataset_json_path']}")
        if summary['hf_dataset_path']:
            print(f"HF Dataset Path: {summary['hf_dataset_path']}")
        print("="*80)
        
        if summary['failed_pdfs']:
            print(f"\nFailed PDFs: {', '.join(summary['failed_pdf_list'])}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
