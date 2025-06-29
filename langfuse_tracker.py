"""
Langfuse integration for tracking pipeline execution and results.
Logs PDF processing, Gemini API calls, and schema validation results.
"""
import os
import logging
import time
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from langfuse import get_client, observe

from schemas import PitchDeckAnalysis, ProcessingResult, SlideContent

logger = logging.getLogger(__name__)


class LangfuseTracker:
    """Handles all Langfuse tracking for the pipeline."""
    
    def __init__(self, 
                 public_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 host: Optional[str] = None):
        """
        Initialize Langfuse tracker.
        
        Args:
            public_key: Langfuse public key (if None, uses LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (if None, uses LANGFUSE_SECRET_KEY env var)
            host: Langfuse host (if None, uses LANGFUSE_HOST env var)
        """
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.langfuse = None
        self.current_session_trace_id = None
        
        if not self.public_key or not self.secret_key:
            logger.warning("Langfuse credentials not found - tracking will be disabled")
            return
        
        try:
            # Initialize Langfuse client using v3 API
            from langfuse import Langfuse
            self.langfuse = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
            logger.info(f"Langfuse tracker initialized with host: {self.host}")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.langfuse = None
    
    def create_session(self, session_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new Langfuse session for tracking a pipeline run.
        In v3, we use traces with session_id metadata instead of separate sessions.
        
        Args:
            session_name: Name for the session
            metadata: Additional metadata for the session
            
        Returns:
            Session ID (generated trace ID for the session)
        """
        if not self.is_enabled():
            session_id = f"disabled_session_{session_name}"
            logger.debug(f"Langfuse disabled - creating mock session: {session_id}")
            return session_id
            
        try:
            # In v3, we create a trace that represents the session
            session_id = f"session_{int(time.time())}_{session_name}"
            
            with self.langfuse.start_as_current_span(
                name=f"Pipeline Session: {session_name}"
            ) as session_span:
                session_span.update_trace(
                    session_id=session_id,
                    metadata=metadata or {},
                    input={"session_name": session_name, "start_time": time.time()}
                )
                # Store the trace ID as our session identifier
                self.current_session_trace_id = session_span.trace_id
                
            logger.info(f"Created Langfuse session trace: {session_name}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create Langfuse session: {str(e)}")
            return f"error_session_{session_name}"
    
    def track_pdf_processing(self, 
                           pdf_filename: str,
                           session_id: str,
                           processing_result: Dict[str, Any]) -> str:
        """
        Track PDF processing results using v3 API.
        
        Args:
            pdf_filename: Name of the PDF file
            session_id: Langfuse session ID
            processing_result: Result from PDF processor
            
        Returns:
            Trace ID
        """
        if not self.is_enabled():
            trace_id = f"disabled_trace_{pdf_filename}"
            logger.debug(f"Langfuse disabled - creating mock trace: {trace_id}")
            return trace_id
            
        try:
            with self.langfuse.start_as_current_span(
                name=f"PDF Processing: {pdf_filename}"
            ) as span:
                span.update_trace(
                    session_id=session_id,
                    input={"pdf_filename": pdf_filename},
                    output=processing_result,
                    metadata={
                        "component": "pdf_processor",
                        "success": processing_result.get("success", False),
                        "slide_count": processing_result.get("slide_count", 0),
                        "processing_time": processing_result.get("processing_time", 0)
                    }
                )
                
                logger.debug(f"Tracked PDF processing for {pdf_filename}")
                return span.trace_id
                
        except Exception as e:
            logger.error(f"Failed to track PDF processing for {pdf_filename}: {str(e)}")
            return f"error_trace_{pdf_filename}"
    
    def track_gemini_analysis(self,
                            pdf_filename: str,
                            session_id: str,
                            image_paths: List[str],
                            analysis_result: Optional[PitchDeckAnalysis],
                            processing_time: float,
                            error_message: Optional[str] = None) -> str:
        """
        Track Gemini API analysis using v3 API.
        
        Args:
            pdf_filename: Name of the PDF file
            session_id: Langfuse session ID
            image_paths: List of image paths sent to Gemini
            analysis_result: Gemini analysis result
            processing_time: Time taken for analysis
            error_message: Error message if analysis failed
            
        Returns:
            Trace ID
        """
        if not self.is_enabled():
            trace_id = f"disabled_trace_{pdf_filename}"
            logger.debug(f"Langfuse disabled - creating mock trace: {trace_id}")
            return trace_id
            
        try:
            # Prepare input and output data
            input_data = {
                "pdf_filename": pdf_filename,
                "image_count": len(image_paths),
                "image_paths": image_paths[:3]  # Limit for readability
            }
            
            if analysis_result:
                output_data = analysis_result.model_dump()
                success = True
            else:
                output_data = {"error": error_message or "Analysis failed"}
                success = False
            
            # Create a generation for the LLM call using v3 API
            with self.langfuse.start_as_current_generation(
                name=f"Gemini Analysis: {pdf_filename}",
                model="gemini-2.0-flash-exp",
                input=input_data,
                model_parameters={"temperature": 0.1}
            ) as gen:
                
                gen.update(output=output_data)
                
                # Update trace with session info
                gen.update_trace(
                    session_id=session_id,
                    metadata={
                        "component": "gemini_analyzer",
                        "success": success,
                        "processing_time": processing_time,
                        "company_name": analysis_result.company_name if analysis_result else None,
                        "industry": analysis_result.industry if analysis_result else None,
                        "total_slides": analysis_result.total_slides if analysis_result else 0
                    },
                    tags=["gemini", "analysis", "llm"]
                )
                
                trace_id = gen.trace_id
                logger.debug(f"Tracked Gemini analysis for {pdf_filename} (trace: {trace_id})")
                return trace_id
                
        except Exception as e:
            logger.error(f"Failed to track Gemini analysis for {pdf_filename}: {e}")
            return f"error_trace_{pdf_filename}"
    
    def track_slide_analysis(self,
                           pdf_filename: str,
                           session_id: str,
                           slide_number: int,
                           slide_analysis: Optional[SlideContent],
                           processing_time: float) -> str:
        """
        Track individual slide analysis using v3 API.
        
        Args:
            pdf_filename: Name of the PDF file
            session_id: Langfuse session ID
            slide_number: Slide number
            slide_analysis: Slide analysis result
            processing_time: Time taken for analysis
            
        Returns:
            Span ID
        """
        if not self.is_enabled():
            span_id = f"disabled_span_{pdf_filename}_slide_{slide_number}"
            logger.debug(f"Langfuse disabled - creating mock span: {span_id}")
            return span_id
            
        try:
            input_data = {
                "pdf_filename": pdf_filename,
                "slide_number": slide_number
            }
            
            if slide_analysis:
                output_data = slide_analysis.model_dump()
                success = True
            else:
                output_data = {"error": "Slide analysis failed"}
                success = False
            
            with self.langfuse.start_as_current_span(
                name=f"Slide Analysis: {pdf_filename} slide {slide_number}"
            ) as span:
                span.update_trace(
                    session_id=session_id,
                    input=input_data,
                    output=output_data,
                    metadata={
                        "component": "slide_analyzer",
                        "success": success,
                        "processing_time": processing_time,
                        "slide_type": slide_analysis.slide_type if slide_analysis else None,
                        "has_charts": slide_analysis.has_charts if slide_analysis else False,
                        "has_images": slide_analysis.has_images if slide_analysis else False
                    }
                )
                
                logger.debug(f"Tracked slide analysis for {pdf_filename} slide {slide_number}")
                return span.span_id
            
        except Exception as e:
            logger.error(f"Failed to track slide analysis: {str(e)}")
            return f"error_span_{pdf_filename}_slide_{slide_number}"
    
    def track_schema_validation(self,
                              pdf_filename: str,
                              session_id: str,
                              validation_success: bool,
                              validation_errors: Optional[List[str]] = None) -> str:
        """
        Track schema validation results using v3 API.
        
        Args:
            pdf_filename: Name of the PDF file
            session_id: Langfuse session ID
            validation_success: Whether validation was successful
            validation_errors: List of validation errors if any
            
        Returns:
            Event ID
        """
        if not self.is_enabled():
            event_id = f"disabled_event_{pdf_filename}_validation"
            logger.debug(f"Langfuse disabled - creating mock event: {event_id}")
            return event_id
            
        try:
            with self.langfuse.start_as_current_span(
                name=f"Schema Validation: {pdf_filename}"
            ) as span:
                span.update_trace(
                    session_id=session_id,
                    input={"pdf_filename": pdf_filename},
                    output={
                        "success": validation_success,
                        "errors": validation_errors or []
                    },
                    metadata={
                        "component": "schema_validator",
                        "success": validation_success,
                        "error_count": len(validation_errors) if validation_errors else 0
                    }
                )
                
                logger.debug(f"Tracked schema validation for {pdf_filename}")
                return span.span_id
            
        except Exception as e:
            logger.error(f"Failed to track schema validation: {str(e)}")
            return f"error_event_{pdf_filename}_validation"
    
    def track_dataset_creation(self,
                             session_id: str,
                             total_entries: int,
                             successful_entries: int,
                             dataset_path: str) -> str:
        """
        Track dataset creation for fine-tuning using v3 API.
        
        Args:
            session_id: Langfuse session ID
            total_entries: Total number of dataset entries
            successful_entries: Number of successful entries
            dataset_path: Path to the created dataset
            
        Returns:
            Trace ID
        """
        if not self.is_enabled():
            trace_id = f"disabled_trace_dataset_creation"
            logger.debug(f"Langfuse disabled - creating mock trace: {trace_id}")
            return trace_id
            
        try:
            with self.langfuse.start_as_current_span(
                name="Dataset Creation"
            ) as span:
                span.update_trace(
                    session_id=session_id,
                    input={
                        "total_entries": total_entries,
                        "dataset_path": dataset_path
                    },
                    output={
                        "successful_entries": successful_entries,
                        "success_rate": successful_entries / total_entries if total_entries > 0 else 0,
                        "dataset_path": dataset_path
                    },
                    metadata={
                        "component": "dataset_creator",
                        "success": successful_entries > 0,
                        "total_entries": total_entries,
                        "successful_entries": successful_entries
                    }
                )
                
                logger.info(f"Tracked dataset creation: {successful_entries}/{total_entries} entries")
                return span.trace_id
            
        except Exception as e:
            logger.error(f"Failed to track dataset creation: {str(e)}")
            return f"error_trace_dataset_creation"
    
    def track_pipeline_summary(self,
                             session_id: str,
                             total_pdfs: int,
                             successful_pdfs: int,
                             total_processing_time: float,
                             summary_stats: Dict[str, Any]) -> str:
        """
        Track overall pipeline execution summary using v3 API.
        
        Args:
            session_id: Langfuse session ID
            total_pdfs: Total number of PDFs processed
            successful_pdfs: Number of successfully processed PDFs
            total_processing_time: Total time for pipeline execution
            summary_stats: Additional summary statistics
            
        Returns:
            Trace ID
        """
        if not self.is_enabled():
            trace_id = f"disabled_trace_pipeline_summary"
            logger.debug(f"Langfuse disabled - creating mock trace: {trace_id}")
            return trace_id
            
        try:
            with self.langfuse.start_as_current_span(
                name="Pipeline Summary"
            ) as span:
                span.update_trace(
                    session_id=session_id,
                    input={
                        "total_pdfs": total_pdfs,
                        "start_time": summary_stats.get("start_time")
                    },
                    output={
                        "successful_pdfs": successful_pdfs,
                        "success_rate": successful_pdfs / total_pdfs if total_pdfs > 0 else 0,
                        "total_processing_time": total_processing_time,
                        "summary_stats": summary_stats
                    },
                    metadata={
                        "component": "pipeline_orchestrator",
                        "success": successful_pdfs > 0,
                        "total_pdfs": total_pdfs,
                        "successful_pdfs": successful_pdfs,
                        "total_processing_time": total_processing_time
                    }
                )
                
                logger.info(f"Tracked pipeline summary: {successful_pdfs}/{total_pdfs} PDFs processed successfully")
                return span.trace_id
            
        except Exception as e:
            logger.error(f"Failed to track pipeline summary: {str(e)}")
            return f"error_trace_pipeline_summary"
    
    def flush(self):
        """Flush all pending traces to Langfuse."""
        if not self.is_enabled():
            logger.debug("Langfuse disabled - skipping flush")
            return
            
        try:
            self.langfuse.flush()
            logger.debug("Flushed all traces to Langfuse")
        except Exception as e:
            logger.error(f"Failed to flush traces to Langfuse: {str(e)}")
    
    def is_enabled(self) -> bool:
        """Check if Langfuse tracking is enabled."""
        return self.langfuse is not None


def setup_langfuse_tracker() -> LangfuseTracker:
    """
    Factory function to create and configure Langfuse tracker.
    
    Returns:
        Configured LangfuseTracker instance
    """
    tracker = LangfuseTracker()
    
    if not tracker.is_enabled():
        logger.warning("Langfuse tracker created but tracking is disabled (missing credentials)")
        return tracker
    
    # Test the connection
    try:
        test_session = tracker.create_session("connection_test", {"test": True})
        logger.info("Langfuse connection test successful")
        tracker.flush()
        return tracker
    except Exception as e:
        logger.error(f"Langfuse connection test failed: {str(e)}")
        # Return disabled tracker instead of raising
        tracker.langfuse = None
        return tracker


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        tracker = setup_langfuse_tracker()
        print("Langfuse tracker ready for use!")
    except Exception as e:
        print(f"Failed to setup Langfuse tracker: {e}")
