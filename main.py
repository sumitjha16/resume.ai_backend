import os
from mistralai import Mistral, SystemMessage, UserMessage
import PyPDF2
from io import BytesIO
import spacy
from typing import Dict, List, Tuple, Optional
import asyncio
import re
from datetime import datetime  # Added datetime import
import json
from collections import defaultdict
import traceback


class EnhancedResumeAnalyzer:
    def __init__(self, mistral_api_key: str):
        """Initialize the Enhanced Resume Analyzer with comprehensive analysis capabilities."""
        self.client = Mistral(api_key=mistral_api_key)

        # Load spaCy model with error handling and progress feedback
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            print("Downloading spaCy model... This might take a few minutes.")
            os.system("python -m spacy download en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")

        # Expanded skill categories for more detailed analysis
        self.skill_categories = {
            'technical_skills': {
                'programming': ['python', 'java', 'javascript', 'c++', 'ruby', 'go'],
                'data': ['sql', 'mongodb', 'postgresql', 'data analysis', 'big data'],
                'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
                'ai_ml': ['machine learning', 'deep learning', 'nlp', 'computer vision']
            },
            'business_skills': {
                'management': ['project management', 'team leadership', 'strategic planning'],
                'analysis': ['business analysis', 'requirements gathering', 'process improvement'],
                'operations': ['operations management', 'supply chain', 'resource planning']
            },
            'soft_skills': {
                'communication': ['presentation', 'writing', 'public speaking'],
                'leadership': ['team building', 'mentoring', 'decision making'],
                'interpersonal': ['collaboration', 'conflict resolution', 'negotiation']
            },
            'domain_specific': {
                'finance': ['financial analysis', 'budgeting', 'forecasting'],
                'marketing': ['digital marketing', 'seo', 'content strategy'],
                'sales': ['sales management', 'account management', 'crm']
            }
        }

        # Common section headers and their variations
        self.section_patterns = {
            'summary': ['summary', 'professional summary', 'profile', 'objective'],
            'experience': ['experience', 'work history', 'employment', 'work experience'],
            'education': ['education', 'academic background', 'qualifications', 'training'],
            'skills': ['skills', 'expertise', 'competencies', 'technical skills'],
            'projects': ['projects', 'key projects', 'portfolio', 'works'],
            'achievements': ['achievements', 'accomplishments', 'awards', 'honors'],
            'certifications': ['certifications', 'certificates', 'licenses'],
            'publications': ['publications', 'research', 'papers'],
            'volunteer': ['volunteer', 'community service', 'social work']
        }

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract and clean text from PDF with enhanced formatting preservation."""
        try:
            # Create BytesIO object from the bytes content
            pdf_file = BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)

            # Enhanced text extraction with formatting preservation
            text_blocks = []
            for page in reader.pages:
                text = page.extract_text()

                # Preserve section breaks and formatting
                text = re.sub(r'(\r\n|\r|\n)\s*(\r\n|\r|\n)', '\n\n', text)
                text = re.sub(r'\s{2,}', ' ', text)
                text_blocks.append(text.strip())

            return '\n\n'.join(text_blocks)
        except Exception as e:
            print(f"Error extracting PDF content: {e}")
            raise

    def identify_section(self, text: str) -> Optional[str]:
        """Identify resume section based on pattern matching."""
        text_lower = text.lower()
        for section, patterns in self.section_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return section
        return None

    def extract_dates(self, text: str) -> List[Tuple[str, str]]:
        """Extract date ranges from text with improved pattern matching."""
        date_patterns = [
            r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
            r'Dec(?:ember)?)[,]?\s+\d{4})\s*(?:-|–|to)\s*(Present|Current|(?:Jan(?:uary)?|'
            r'Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
            r'Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,]?\s+\d{4})',
            r'(\d{4})\s*(?:-|–|to)\s*(Present|Current|\d{4})'
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start_date, end_date = match.groups()
                dates.append((start_date.strip(), end_date.strip()))
        return dates

    def extract_metrics(self, text: str) -> Dict[str, List[str]]:
        """Extract and categorize quantifiable achievements."""
        metrics = defaultdict(list)

        patterns = {
            'improvements': [
                r'(?:increased|improved|enhanced|grew|raised|boosted)\s+(?:by\s+)?(\d+[%+]|\$[\d,]+K?M?|[\d,]+\s+(?:percent|users|customers|clients|members|dollars|sales))',
                r'(\d+[%+]|\$[\d,]+K?M?)\s+(?:increase|improvement|growth|boost|gain)'
            ],
            'reductions': [
                r'(?:reduced|decreased|cut|lowered)\s+(?:by\s+)?(\d+[%+]|\$[\d,]+K?M?|[\d,]+\s+(?:percent|costs|expenses|time|hours))',
                r'(\d+[%+]|\$[\d,]+K?M?)\s+(?:reduction|decrease|cut|savings)'
            ],
            'achievements': [
                r'(?:achieved|accomplished|delivered|generated|produced|managed)\s+(?:revenue\s+of\s+)?(\$[\d,]+K?M?|\d+[%+]|\d+\s+(?:users|customers|projects|implementations))',
                r'(?:led|supervised|managed|coordinated)\s+(?:team\s+of\s+)?(\d+\+?\s+(?:people|employees|members|developers))'
            ]
        }

        for category, category_patterns in patterns.items():
            for pattern in category_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                metrics[category].extend(match.group(1) for match in matches)

        return dict(metrics)

    def categorize_skills(self, text: str) -> Dict[str, Dict[str, List[str]]]:
        """Categorize skills with hierarchical classification."""
        doc = self.nlp(text.lower())
        found_skills = defaultdict(lambda: defaultdict(set))

        # Extract skills using NLP and pattern matching
        for main_category, subcategories in self.skill_categories.items():
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    if keyword in text.lower():
                        # Find the complete phrase containing the keyword
                        matches = re.finditer(
                            rf'\b{keyword}\b(?:[,\s]+(?:\w+\s+){0, 3}\w+)*',
                            text.lower()
                        )
                        for match in matches:
                            found_skills[main_category][subcategory].add(match.group().strip())

        # Convert sets to sorted lists for JSON serialization
        return {
            category: {
                subcat: sorted(list(skills))
                for subcat, skills in subcategories.items()
            }
            for category, subcategories in found_skills.items()
        }

    def process_resume_content(self, text: str) -> Dict[str, any]:
        """Process resume content with enhanced section detection and analysis."""
        sections = defaultdict(list)
        current_section = None
        section_text = []

        # Split text into lines for processing
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line is a section header
            detected_section = self.identify_section(line)

            if detected_section:
                # Save previous section content
                if current_section and section_text:
                    sections[current_section].append('\n'.join(section_text))
                    section_text = []
                current_section = detected_section
            elif current_section:
                section_text.append(line)

        # Add last section
        if current_section and section_text:
            sections[current_section].append('\n'.join(section_text))

        # Process sections for detailed analysis
        processed_content = {
            'raw_text': text,  # Store complete resume text
            'sections': dict(sections),
            'skills': self.categorize_skills(text),
            'metrics': self.extract_metrics(text),
            'dates': self.extract_dates(text),
            'section_statistics': {
                section: {
                    'word_count': len(' '.join(content).split()),
                    'sentence_count': len(list(self.nlp(' '.join(content)).sents))
                }
                for section, content in sections.items()
            }
        }

        return processed_content

    # main.py - Updated get_ai_analysis method
    async def get_ai_analysis(self, resume_content: Dict[str, any]) -> Dict[str, str]:
        """Generate comprehensive AI analysis with improved prompts for deeper insights"""
        try:
            if not resume_content or not isinstance(resume_content, dict):
                raise ValueError("Invalid resume content provided")

            # Validate required content
            required_keys = ['raw_text', 'skills', 'metrics', 'dates', 'section_statistics']
            missing_keys = [key for key in required_keys if key not in resume_content]
            if missing_keys:
                raise ValueError(f"Missing required resume content: {', '.join(missing_keys)}")

            analyses = {
                'career_trajectory': '',
                'skills_analysis': '',
                'resume_optimization': '',
                'action_plan': ''
            }

            analysis_prompts = {
                'career_trajectory': {
                    'prompt': """Analyze the career trajectory based on the provided resume data. Focus on:
                    1. Career progression pattern
                    2. Key achievements and milestones
                    3. Industry transitions and adaptability
                    4. Leadership growth
                    5. Future career potential

                    Provide a detailed analysis with specific examples from the resume.""",
                    'timeout': 45.0
                },
                'skills_analysis': {
                    'prompt': """Analyze the technical and professional skills presented in the resume. Focus on:
                    1. Core technical competencies
                    2. Skill relevance to current market
                    3. Skill gaps and improvement areas
                    4. Industry-specific expertise
                    5. Transferable skills

                    Provide specific examples and market context for the analysis.""",
                    'timeout': 45.0
                },
                'resume_optimization': {
                    'prompt': """Provide specific recommendations for resume optimization. Focus on:
                    1. Content improvement opportunities
                    2. Achievement quantification
                    3. Key selling points enhancement
                    4. Format and structure suggestions
                    5. ATS optimization tips

                    Offer actionable suggestions with examples.""",
                    'timeout': 45.0
                },
                'action_plan': {
                    'prompt': """Create a detailed action plan for professional development. Include:
                    1. Short-term goals (0-6 months)
                    2. Medium-term goals (6-18 months)
                    3. Skill development priorities
                    4. Networking strategies
                    5. Career advancement steps

                    Provide specific, actionable steps with timelines.""",
                    'timeout': 45.0
                }
            }

            for analysis_type, config in analysis_prompts.items():
                max_retries = 2
                retry_count = 0

                while retry_count <= max_retries:
                    try:
                        # Prepare resume summary with key information
                        resume_summary = {
                            'text': resume_content['raw_text'][:2000],
                            'skills': resume_content.get('skills', {}),
                            'metrics': resume_content.get('metrics', {}),
                            'dates': resume_content.get('dates', [])
                        }

                        # Create system message for analysis context
                        system_message = """You are an expert career advisor and resume analyst. 
                        Provide detailed, actionable insights based on the resume content.
                        Focus on specific examples and concrete recommendations.
                        Format your response in clear paragraphs with line breaks between main points."""

                        # Prepare messages for the AI model
                        messages = [
                            SystemMessage(content=system_message),
                            UserMessage(content=f"""Analyze this professional profile:

                            Resume Content:
                            {resume_summary['text']}

                            Professional Skills:
                            {json.dumps(resume_summary['skills'], indent=2)}

                            Career Timeline:
                            {json.dumps(resume_summary['dates'], indent=2)}

                            Key Metrics:
                            {json.dumps(resume_summary['metrics'], indent=2)}

                            Analysis Request:
                            {config['prompt']}

                            Format your response in clear paragraphs with line breaks between main points.""")
                        ]

                        # Get response from the AI model
                        response = await asyncio.wait_for(
                            self.client.chat.complete_async(
                                model="mistral-medium",
                                messages=messages,
                                temperature=0.7,
                                max_tokens=1000
                            ),
                            timeout=config['timeout']
                        )

                        # Store the response
                        analyses[analysis_type] = response.choices[0].message.content
                        break

                    except asyncio.TimeoutError:
                        retry_count += 1
                        if retry_count <= max_retries:
                            print(f"Timeout in {analysis_type} analysis, attempt {retry_count}/{max_retries}")
                            await asyncio.sleep(1)
                        else:
                            print(f"All retries failed for {analysis_type} analysis")
                            analyses[
                                analysis_type] = "Analysis could not be completed due to timeout. Please try again."

                    except Exception as e:
                        print(f"Error in {analysis_type} analysis: {str(e)}\n{traceback.format_exc()}")
                        analyses[analysis_type] = f"Analysis encountered an error: {str(e)}"
                        break

            if not any(analyses.values()):
                raise ValueError("No analyses could be completed")

            return {
                "analysis": analyses,
                "extracted_content": resume_content
            }

        except Exception as e:
            print(f"Error in AI analysis: {str(e)}\n{traceback.format_exc()}")
            raise

    async def analyze_resume(self, pdf_path: str) -> Dict[str, any]:
        """Perform comprehensive resume analysis with detailed insights."""
        try:
            # Read and process PDF
            with open(pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()

            # Extract and process content
            raw_text = self.extract_text_from_pdf(pdf_content)
            processed_content = self.process_resume_content(raw_text)

            # Get AI analysis
            analysis = await self.get_ai_analysis(processed_content)

            # Combine results
            return {
                "analysis": analysis,
                "extracted_content": processed_content,
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0"
            }

        except Exception as e:
            print(f"Error in resume analysis: {e}")
            raise


async def main():
    """Main function with enhanced error handling and output formatting."""
    try:
        mistral_api_key = "T7nrkTxlxs6lpdB0Jpr0poZwcBwgcWwF"
        if not mistral_api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")

        analyzer = EnhancedResumeAnalyzer(mistral_api_key)
        pdf_path = "Resume (1).pdf"

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Resume file not found: {pdf_path}")

        print("\nAnalyzing resume... This may take a few moments.\n")
        results = await analyzer.analyze_resume(pdf_path)

        # Print structured analysis results
        print("\n=== Career Development Analysis ===\n")
        print(results["analysis"]["career_trajectory"])

        print("\n=== Skills Assessment ===\n")
        print(results["analysis"]["skills_analysis"])

        print("\n=== Resume Optimization Recommendations ===\n")
        print(results["analysis"]["resume_optimization"])

        print("\n=== Action Plan ===\n")
        print(results["analysis"]["action_plan"])

        # Save results to file
        output_file = f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed analysis saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())