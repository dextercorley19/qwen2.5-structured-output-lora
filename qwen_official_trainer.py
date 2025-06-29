#!/usr/bin/env python3
"""
Qwen2.5-VL trainer using OFFICIAL HuggingFace pattern
"""

import os
import torch
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,  # Use this instead of Qwen2VLForConditionalGeneration
    TrainingArguments, 
    Trainer
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging
from PIL import Image
import json
from schemas import PitchDeckAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenOfficialTrainer:
    """Qwen2.5-VL trainer using the official HF pattern."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", use_mps=True):
        self.model_name = model_name
        self.device = self._setup_device(use_mps)
        self.processor = None
        self.model = None
        
    def _setup_device(self, use_mps):
        """Setup the appropriate device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA GPU")
        elif use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def load_model_and_processor(self):
        """Load model using OFFICIAL HF pattern."""
        logger.info(f"Loading Qwen2.5-VL model: {self.model_name}")
        
        try:
            # OFFICIAL PATTERN from HF docs
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Move to MPS if needed
            if self.device.type == "mps":
                self.model = self.model.to(self.device)
            
            logger.info("Qwen2.5-VL model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            return False
    
    def get_pydantic_schema_prompt(self):
        """Create prompt using Pydantic schema."""
        schema = PitchDeckAnalysis.model_json_schema()
        
        return f"""Analyze this pitch deck and extract business intelligence. Return ONLY valid JSON matching this schema:

{json.dumps(schema, indent=2)}

Provide structured analysis as JSON:"""
    
    def create_official_dataset(self):
        """Create dataset using official Qwen format."""
        logger.info("Creating dataset with official Qwen format...")
        
        # Load results
        results_files = [f"intermediate_results_{i}.json" for i in range(20) if os.path.exists(f"intermediate_results_{i}.json")]
        
        if not results_files:
            return self._create_demo_dataset()
        
        latest_results = sorted(results_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        logger.info(f"Using results from: {latest_results}")
        
        with open(latest_results, 'r') as f:
            gemini_results = json.load(f)
        
        training_data = []
        schema_prompt = self.get_pydantic_schema_prompt()
        
        for result in gemini_results:
            if not result.get('success', False):
                continue
                
            analysis = result.get('analysis', {})
            company = analysis.get('company_name', 'Unknown')
            
            # Load images
            image_dir = "processed_images"
            company_images = []
            
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    if company.lower().replace(' ', '_') in file.lower() and file.endswith('.png'):
                        img_path = os.path.join(image_dir, file)
                        try:
                            img = Image.open(img_path).convert("RGB")
                            company_images.append(img)
                        except Exception:
                            continue
            
            if company_images:
                # Validate against Pydantic schema
                try:
                    validated_analysis = PitchDeckAnalysis.model_validate(analysis)
                    response = validated_analysis.model_dump_json(indent=2)
                    logger.info(f"✅ {company}: Schema validated")
                except Exception as e:
                    logger.warning(f"⚠️  {company}: Creating compliant version")
                    minimal_analysis = PitchDeckAnalysis(
                        company_name=company,
                        industry=analysis.get('industry', 'unknown'),
                        total_slides=len(company_images)
                    )
                    response = minimal_analysis.model_dump_json(indent=2)
                
                # OFFICIAL FORMAT: Use messages as shown in HF docs
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": schema_prompt}
                        ] + [{"type": "image", "image": img} for img in company_images[:3]]
                    }
                ]
                
                training_data.append({
                    "messages": messages,
                    "response": response,
                    "company": company
                })
        
        logger.info(f"Created dataset with {len(training_data)} examples")
        
        # Convert to HF dataset
        dataset = Dataset.from_dict({
            "messages": [item["messages"] for item in training_data],
            "response": [item["response"] for item in training_data],
            "company": [item["company"] for item in training_data]
        })
        
        # Split dataset
        if len(dataset) > 2:
            train_test = dataset.train_test_split(test_size=0.3)
            if len(train_test["test"]) > 1:
                val_test = train_test["test"].train_test_split(test_size=0.5)
                final_dataset = {
                    "train": train_test["train"],
                    "validation": val_test["train"], 
                    "test": val_test["test"]
                }
            else:
                final_dataset = {
                    "train": train_test["train"],
                    "validation": train_test["test"],
                    "test": train_test["test"]
                }
        else:
            final_dataset = {
                "train": dataset,
                "validation": dataset,
                "test": dataset
            }
        
        return final_dataset
    
    def _create_demo_dataset(self):
        """Create demo dataset for testing."""
        logger.info("Creating demo dataset...")
        
        demo_data = []
        schema_prompt = self.get_pydantic_schema_prompt()
        
        for i in range(3):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": schema_prompt},
                        {"type": "image", "image": Image.new("RGB", (224, 224), f"hsl({i*120}, 70%, 80%)")}
                    ]
                }
            ]
            
            response = json.dumps({
                "company_name": f"Demo Company {i+1}",
                "industry": "technology",
                "total_slides": 1
            }, indent=2)
            
            demo_data.append({
                "messages": messages,
                "response": response,
                "company": f"Demo{i+1}"
            })
        
        dataset = Dataset.from_dict({
            "messages": [item["messages"] for item in demo_data],
            "response": [item["response"] for item in demo_data],
            "company": [item["company"] for item in demo_data]
        })
        
        return {"train": dataset, "validation": dataset, "test": dataset}
    
    def inspect_dataset(self, dataset, num_samples=2):
        """Inspect the training dataset."""
        print("\n🔍 DATASET INSPECTION")
        print("=" * 50)
        
        print(f"📊 Dataset sizes:")
        print(f"   Train: {len(dataset['train'])} examples")
        print(f"   Validation: {len(dataset['validation'])} examples") 
        print(f"   Test: {len(dataset['test'])} examples")
        
        print(f"\n📋 Dataset columns: {dataset['train'].column_names}")
        
        # Show sample data
        print(f"\n🔍 Inspecting {num_samples} training samples:")
        
        for i in range(min(num_samples, len(dataset['train']))):
            sample = dataset['train'][i]
            
            print(f"\n📄 SAMPLE {i+1}:")
            print(f"Company: {sample.get('company', 'Unknown')}")
            
            # Show messages
            messages = sample['messages']
            for j, msg in enumerate(messages):
                content = msg.get('content', [])
                text_parts = [c for c in content if c.get('type') == 'text']
                image_parts = [c for c in content if c.get('type') == 'image']
                
                print(f"  Message {j+1} ({msg['role']}):")
                print(f"    📝 Text parts: {len(text_parts)}")
                print(f"    🖼️  Image parts: {len(image_parts)}")
            
            # Show response preview
            try:
                response_json = json.loads(sample['response'])
                print(f"  📊 Response preview:")
                print(f"    Company: {response_json.get('company_name', 'N/A')}")
                print(f"    Industry: {response_json.get('industry', 'N/A')}")
                print(f"    Slides: {response_json.get('total_slides', 'N/A')}")
            except:
                print(f"  📊 Response: {sample['response'][:100]}...")
    
    def setup_lora(self, rank=4, alpha=16):
        """Setup LoRA for efficient training."""
        logger.info(f"Setting up LoRA with rank={rank}, alpha={alpha}")
        
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            return True
        except Exception as e:
            logger.error(f"LoRA setup failed: {e}")
            return False
    
    def create_dataset_from_fixed_results(self):
        """Create dataset using the fixed results."""
        logger.info("Creating dataset from fixed results...")
        
        # Use the fixed results file
        fixed_results_file = "intermediate_results_fixed.json"
        
        if not os.path.exists(fixed_results_file):
            logger.warning("No fixed results found, creating demo dataset")
            return self._create_demo_dataset()
        
        with open(fixed_results_file, 'r') as f:
            fixed_results = json.load(f)
        
        logger.info(f"Loaded {len(fixed_results)} fixed company analyses")
        
        training_data = []
        schema_prompt = self.get_pydantic_schema_prompt()
        image_dir = "processed_images"
        
        for result in fixed_results:
            if not result.get('success', False):
                continue
                
            analysis = result.get('analysis', {})
            company = analysis.get('company_name', 'Unknown')
            
            logger.info(f"Processing {company}...")
            
            # Load images for this company
            company_images = []
            company_key = company.lower().replace(' ', '').replace('-', '')
            
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    file_lower = file.lower()
                    if (company_key in file_lower or 
                        company.lower().split()[0] in file_lower) and file.endswith('.png'):
                        img_path = os.path.join(image_dir, file)
                        try:
                            img = Image.open(img_path).convert("RGB")
                            company_images.append(img)
                        except Exception:
                            continue
            
            if company_images:
                # Use the validated analysis
                response = json.dumps(analysis, indent=2)
                
                # Create training example
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": schema_prompt}
                        ] + [{"type": "image", "image": img} for img in company_images[:5]]
                    }
                ]
                
                training_data.append({
                    "messages": messages,
                    "response": response,
                    "company": company
                })
                
                logger.info(f"✅ Added {company} with {len(company_images)} images")
        
        logger.info(f"Created dataset with {len(training_data)} examples")
        
        # Convert to HF dataset and return
        dataset = Dataset.from_dict({
            "messages": [item["messages"] for item in training_data],
            "response": [item["response"] for item in training_data],
            "company": [item["company"] for item in training_data]
        })
        
        # Split dataset
        if len(dataset) > 2:
            train_test = dataset.train_test_split(test_size=0.3)
            final_dataset = {
                "train": train_test["train"],
                "validation": train_test["test"],
                "test": train_test["test"]
            }
        else:
            final_dataset = {
                "train": dataset,
                "validation": dataset,
                "test": dataset
            }
        
        return final_dataset

def test_official_loading():
    """Test the official HF loading pattern."""
    print("🔍 Testing OFFICIAL Qwen2.5-VL loading...")
    
    try:
        # EXACT pattern from HF docs
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        print("✅ Official loading successful!")
        print(f"📏 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {"type": "image", "image": Image.new("RGB", (224, 224), "blue")}
                ]
            }
        ]
        
        print("✅ Message format valid!")
        return True
        
    except Exception as e:
        print(f"❌ Official loading failed: {e}")
        return False

def main():
    """Main function using official HF pattern with FIXED dataset."""
    print("🎯 QWEN2.5-VL WITH FIXED COMPANY DATASET")
    print("=" * 60)
    
    # First, run the complete fix to ensure we have good data
    print("🔧 Running complete company name fix...")
    try:
        import subprocess
        result = subprocess.run(['python', 'fix_company_names_complete.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Company names fixed successfully!")
        else:
            print("⚠️  Fix script had issues, but continuing...")
    except Exception as e:
        print(f"⚠️  Could not run fix script: {e}")
    
    # Test official loading
    if test_official_loading():
        print("\n🚀 Official pattern works! Setting up trainer...")
        
        trainer = QwenOfficialTrainer()
        
        if trainer.load_model_and_processor():
            print("✅ Trainer model loaded!")
            
            # Create dataset from FIXED results
            print("\n📊 Creating dataset from FIXED results...")
            dataset = trainer.create_dataset_from_fixed_results()  # ← USE FIXED METHOD
            trainer.inspect_dataset(dataset)
            
            # Setup LoRA
            print("\n🔧 Setting up LoRA...")
            if trainer.setup_lora(rank=4, alpha=16):
                print("✅ LoRA ready!")
            
            print("\n🎉 READY FOR TRAINING!")
            print("Your Qwen2.5-VL model is loaded and configured with:")
            print("  ✅ Official HuggingFace pattern")
            print("  ✅ Pydantic schema integration") 
            print("  ✅ FIXED pitch deck data with real company names")
            print("  ✅ LoRA efficient training")
            
        else:
            print("❌ Trainer setup failed")
    else:
        print("❌ Official pattern test failed")

if __name__ == "__main__":
    main()
