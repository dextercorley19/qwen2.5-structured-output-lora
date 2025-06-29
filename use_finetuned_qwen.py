#!/usr/bin/env python3
"""
Use the fine-tuned Qwen2.5-VL model for pitch deck analysis
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
import json
from schemas import PitchDeckAnalysis

class FineTunedQwenAnalyzer:
    """Use the fine-tuned Qwen model for pitch deck analysis."""
    
    def __init__(self, base_model="Qwen/Qwen2.5-VL-3B-Instruct", lora_path="./qwen_pitch_deck_lora"):
        self.processor = AutoProcessor.from_pretrained(base_model)
        
        # Load base model
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load fine-tuned LoRA weights
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        self.model.eval()
        
        print(f"✅ Fine-tuned Qwen model loaded from {lora_path}")
    
    def analyze_pitch_deck(self, images, company_name=None):
        """Analyze a pitch deck with multiple images."""
        
        # Create the structured prompt
        prompt = self.get_analysis_prompt()
        
        # Prepare messages in Qwen format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [{"type": "image", "image": img} for img in images[:10]]  # Max 10 images
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            # Parse and validate
            analysis_dict = json.loads(json_str)
            validated_analysis = PitchDeckAnalysis.model_validate(analysis_dict)
            
            return {
                "success": True,
                "analysis": validated_analysis,
                "raw_response": response
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_response": response
            }
    
    def get_analysis_prompt(self):
        """Get the structured analysis prompt."""
        schema = PitchDeckAnalysis.model_json_schema()
        
        return f"""Analyze this pitch deck and extract business intelligence. Return ONLY valid JSON matching this schema:

{json.dumps(schema, indent=2)}

Provide structured analysis as JSON:"""

def analyze_new_pitch_deck(image_paths):
    """Analyze a new pitch deck from image paths."""
    print("🔍 ANALYZING NEW PITCH DECK WITH FINE-TUNED QWEN")
    print("=" * 60)
    
    # Load images
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            print(f"✅ Loaded: {img_path}")
        except Exception as e:
            print(f"❌ Failed to load {img_path}: {e}")
    
    if not images:
        print("❌ No images loaded!")
        return None
    
    # Load fine-tuned model
    analyzer = FineTunedQwenAnalyzer()
    
    # Analyze
    print(f"\n🤖 Analyzing {len(images)} slides...")
    result = analyzer.analyze_pitch_deck(images)
    
    if result["success"]:
        analysis = result["analysis"]
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📊 Company: {analysis.company_name}")
        print(f"🏭 Industry: {analysis.industry}")
        print(f"📄 Total Slides: {analysis.total_slides}")
        print(f"⭐ Quality: {analysis.deck_quality}")
        
        if analysis.notes:
            print(f"📝 Notes: {analysis.notes}")
        
        return analysis
    else:
        print(f"❌ Analysis failed: {result['error']}")
        print(f"Raw response: {result['raw_response'][:500]}...")
        return None

def main():
    """Example usage."""
    # Example: Analyze images from a directory
    import os
    
    sample_images = []
    if os.path.exists("processed_images"):
        # Get first few Airbnb images as example
        airbnb_images = [f for f in os.listdir("processed_images") if "airbnb" in f.lower()][:5]
        sample_images = [os.path.join("processed_images", f) for f in airbnb_images]
    
    if sample_images:
        analysis = analyze_new_pitch_deck(sample_images)
    else:
        print("No sample images found. Place pitch deck images in a folder and update the paths.")

if __name__ == "__main__":
    main()
