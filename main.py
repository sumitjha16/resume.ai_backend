import os
from mistralai import Mistral, SystemMessage, UserMessage
import PyPDF2
from io import BytesIO
import re
from datetime import datetime
import json
from collections import defaultdict
import asyncio
import traceback
from typing import Dict, List, Tuple, Optional, Any


class EnhancedResumeAnalyzer:
    def __init__(self, mistral_api_key: str):
        """Initialize the Enhanced Resume Analyzer with analysis capabilities."""
        self.client = Mistral(api_key=mistral_api_key)

        # Skill categories for detailed analysis
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

        # Section patterns for resume parsing
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
        """Extract text from PDF while preserving formatting."""
        try:
            pdf_file = BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text_blocks = []
            for page in reader.pages:
                text = page.extract_text()
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

    def count_sentences(self, text: str) -> int:
        """Count sentences in text using regex."""
        return len(re.split(r'[.!?]+', text))

    def extract_dates(self, text: str) -> List[str]:
        """Extract dates from text using regex patterns."""
        # Date patterns (add more patterns as needed)
        date_patterns = [
            r'\b\d{4}\b',  # Year
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}'
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        return sorted(list(set(dates)))

    def extract_metrics(self, text: str) -> List[str]:
        """Extract metrics and numerical achievements."""
        # Patterns for metrics (percentages, currencies, quantities)
        metric_patterns = [
            r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|k))?',
            r'\d+(?:,\d{3})*%',
            r'\d+(?:,\d{3})*\+?\s*(?:users|customers|clients|employees|people|projects|years)'
        ]
        
        metrics = []
        for pattern in metric_patterns:
            metrics.extend(re.findall(pattern, text, re.IGNORECASE))
        return sorted(list(set(metrics)))

    def categorize_skills(self, text: str) -> dict:
        """Categorize skills from text using pattern matching."""
        text_lower = text.lower()
        found_skills = defaultdict(lambda: defaultdict(set))
        
        for main_category, subcategories in self.skill_categories.items():
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    pattern = rf'\b{re.escape(keyword)}\b(?:[,\s]+(?:\w+\s+){{0,3}}\w+)*'
                    matches = re.finditer(pattern, text_lower)
                    for match in matches:
                        found_skills[main_category][subcategory].add(match.group().strip())
        
        return {
            category: {
                subcat: sorted(list(skills))
                for subcat, skills in subcategories.items()
            }
            for category, subcategories in found_skills.items()
        }

    def process_resume_content(self, text: str) -> dict:
        """Process resume content with section detection and analysis."""
        sections = defaultdict(list)
        current_section = None
        section_text = []

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            detected_section = self.identify_section(line)
            if detected_section:
                if current_section and section_text:
                    sections[current_section].append('\n'.join(section_text))
                    section_text = []
                current_section = detected_section
            elif current_section:
                section_text.append(line)

        if current_section and section_text:
            sections[current_section].append('\n'.join(section_text))

        return {
            'raw_text': text,
            'sections': dict(sections),
            'skills': self.categorize_skills(text),
            'metrics': self.extract_metrics(text),
            'dates': self.extract_dates(text),
            'section_statistics': {
                section: {
                    'word_count': len(' '.join(content).split()),
                    'sentence_count': self.count_sentences(' '.join(content))
                }
                for section, content in sections.items()
            }
        }

    async def get_ai_analysis(self, resume_content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI analysis of resume content."""
        try:
            system_prompt = """You are an expert career advisor and resume analyst. Your task is to provide comprehensive, actionable feedback on the resume content provided. Focus on specific examples and practical recommendations. Please analyze the following areas:

1. Career Progression: Evaluate the career trajectory, noting key transitions and growth areas.
2. Skills Assessment: Analyze technical and soft skills, identifying strengths and gaps.
3. Resume Impact: Suggest improvements for content, structure, and presentation.
4. Future Opportunities: Recommend next career steps and skill development areas.

Format your response in clear sections with specific examples and actionable recommendations."""

            resume_text = resume_content['raw_text'][:3000]  # Limit text length
            skills_summary = json.dumps(resume_content['skills'], indent=2)
            metrics_summary = ', '.join(resume_content['metrics'])

            analysis_prompt = f"""Please analyze this professional profile:

Resume Content:
{resume_text}

Skills Overview:
{skills_summary}

Key Metrics and Achievements:
{metrics_summary}

Please provide detailed analysis for each of the areas mentioned in the system prompt."""

            messages = [
                SystemMessage(content=system_prompt),
                UserMessage(content=analysis_prompt)
            ]

            response = await self.client.chat.complete_async(
                model="mistral-medium",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )

            analysis_text = response.choices[0].message.content

            # Structure the analysis response
            return {
                "analysis": {
                    "complete_analysis": analysis_text,
                    "timestamp": datetime.now().isoformat()
                },
                "extracted_content": resume_content
            }

        except Exception as e:
            print(f"Error in AI analysis: {str(e)}\n{traceback.format_exc()}")
            raise

    async def analyze_resume(self, pdf_path: str) -> Dict[str, Any]:
        """Perform complete resume analysis."""
        try:
            # Read PDF file
            with open(pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()

            # Extract and process text
            raw_text = self.extract_text_from_pdf(pdf_content)
            processed_content = self.process_resume_content(raw_text)

            # Get AI analysis
            analysis_result = await self.get_ai_analysis(processed_content)

            return {
                "analysis": analysis_result["analysis"],
                "extracted_content": processed_content,
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0"
            }

        except Exception as e:
            print(f"Error in resume analysis: {e}")
            raise


async def main():
    """Main function to run the resume analyzer."""
    try:
        # Initialize with your Mistral API key
        mistral_api_key = "YOUR_MISTRAL_API_KEY"  # Replace with your actual API key
        analyzer = EnhancedResumeAnalyzer(mistral_api_key)
        
        # Specify the path to your PDF resume
        pdf_path = "Resume.pdf"  # Replace with your PDF path
        
        print("\nAnalyzing resume... Please wait...\n")
        results = await analyzer.analyze_resume(pdf_path)
        
        # Print analysis results
        print("\n=== Resume Analysis Results ===\n")
        print(results["analysis"]["complete_analysis"])
        
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
