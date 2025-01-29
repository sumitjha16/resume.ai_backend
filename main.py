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
        """Initialize the Enhanced Resume Analyzer with lightweight analysis capabilities."""
        self.client = Mistral(api_key=mistral_api_key)

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

    def identify_section(self, text: str) -> str:
        """Identify resume section based on pattern matching."""
        text_lower = text.lower()
        for section, patterns in self.section_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return section
        return None

    def count_sentences(self, text: str) -> int:
        """Simple sentence counter using regular expressions."""
        return len(re.split(r'[.!?]+', text))

    def extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        date_patterns = [
            r'\b\d{4}\b',
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}'
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        return sorted(list(set(dates)))

    def extract_metrics(self, text: str) -> List[str]:
        """Extract metrics and achievements with numbers."""
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
        """Categorize skills using regex pattern matching."""
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
        """Process resume content with lightweight section detection and analysis."""
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

        processed_content = {
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

        return processed_content

    async def get_ai_analysis(self, resume_content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive AI analysis with improved prompts for deeper insights"""
        try:
            if not resume_content or not isinstance(resume_content, dict):
                raise ValueError("Invalid resume content provided")

            analyses = {
                'career_trajectory': '',
                'skills_analysis': '',
                'resume_optimization': '',
                'action_plan': ''
            }
            analysis_prompts = {
                'career_trajectory': {
                    'prompt': """Analyze the career trajectory based on the provided resume data:
                            1. Career progression pattern
        - Track job titles and promotions timeline
        - Note major role transitions

        2. Key achievements
        - List quantified accomplishments
        - Highlight awards received

        3. Industry transitions
        - Document industry changes
        - Note adaptation success

        4. Leadership growth
        - Track team size managed
        - Note scope of responsibility

        5. Future potential
        - Identify next career move
        - Assess growth opportunities""",
        'timeout': 45.0
                },
                'skills_analysis': {
                    'prompt': """Analyze the technical and professional skills:
        1. Core competencies
        - List main technical skills
        - Note proficiency levels

        2. Market relevance
        - Match skills to job requirements
        - Identify high-demand abilities

        3. Skill gaps
        - List missing critical skills
        - Suggest needed certifications

        4. Industry expertise
        - Note specialized knowledge
        - List domain experience

        5. Transferable skills
        - Identify cross-industry skills
        - List universal abilities""",
                    'timeout': 45.0
                },
                'resume_optimization': {
                    'prompt': """Optimization recommendations:
        1. Content improvements
        - Add missing metrics
        - Strengthen examples

        2. Quantification
        - Add specific numbers
        - Include scope details

        3. Key selling points
        - Highlight unique skills
        - Emphasize achievements

        4. Format suggestions
        - Improve readability
        - Enhance organization

        5. ATS optimization
        - Add relevant keywords
        - Adjust formatting""",
                    'timeout': 45.0
                },
                'action_plan': {
                    'prompt': """Action plan:
        1. Short-term goals
        - List 3-month priorities
        - Set immediate targets

        2. Medium-term goals
        - Define 1-year objectives
        - Plan major milestones

        3. Skill priorities
        - List skills to acquire
        - Identify resources

        4. Networking
        - Target key events
        - Plan connections

        5. Career steps
        - Set promotion goals
        - List target companies""",
                    'timeout': 45.0
                }
            }




            for analysis_type, config in analysis_prompts.items():
                max_retries = 2
                retry_count = 0

                while retry_count <= max_retries:
                    try:
                        resume_summary = {
                            'text': resume_content['raw_text'][:2000],
                            'skills': resume_content.get('skills', {}),
                            'metrics': resume_content.get('metrics', {}),
                            'dates': resume_content.get('dates', [])
                        }

                        system_message = """You are an expert career advisor and resume analyst. 
                        Provide detailed, actionable insights based on the resume content.
                        Focus on specific examples and concrete recommendations.
                        Format your response in clear paragraphs with line breaks between main points."""

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

                        response = await asyncio.wait_for(
                            self.client.chat.complete_async(
                                model="mistral-medium",
                                messages=messages,
                                temperature=0.7,
                                max_tokens=1000
                            ),
                            timeout=config['timeout']
                        )

                        analyses[analysis_type] = response.choices[0].message.content
                        break

                    except asyncio.TimeoutError:
                        retry_count += 1
                        if retry_count <= max_retries:
                            print(f"Timeout in {analysis_type} analysis, attempt {retry_count}/{max_retries}")
                            await asyncio.sleep(1)
                        else:
                            print(f"All retries failed for {analysis_type} analysis")
                            analyses[analysis_type] = "Analysis could not be completed due to timeout. Please try again."

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

    async def analyze_resume(self, pdf_path: str) -> Dict[str, Any]:
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
                "analysis": analysis["analysis"],
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
        mistral_api_key = "YOUR_MISTRAL_API_KEY"  # Replace with your actual API key
        if not mistral_api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")

        analyzer = EnhancedResumeAnalyzer(mistral_api_key)
        pdf_path = "Resume.pdf"  # Replace with your PDF path

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
