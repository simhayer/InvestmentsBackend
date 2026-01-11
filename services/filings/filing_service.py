import requests
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sec_parser as sp
from services.finnhub.finnhub_service import FinnhubService
from services.vector.vector_store_service import VectorStoreService
from sqlalchemy.orm import Session

class FilingService:
    def __init__(self):
        self.client = FinnhubService().get_finnhub_client()
        # Good job on the User-Agent; it is mandatory for SEC access.
        self.headers = {'User-Agent': 'WALLSTREETAI hayersimrat23@gmail.com'}
        self.session = requests.Session() # Better performance for multiple calls

    def get_filing_metadata(self, symbol: str, years: int = 3) -> List[Dict[str, Any]]:
        """Fetch filtered 10-K and 10-Q metadata."""
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        
        try:
            all_filings = self.client.filings(symbol=symbol, _from=start_date)
            target_forms = {'10-K', '10-Q'}
            return [f for f in all_filings if f.get('form') in target_forms]
        except Exception as e:
            print(f"Finnhub Error: {e}")
            return []

    def download_filing_content(self, report_url: str) -> str:
        """Downloads HTML content, handling iXBRL redirect layers."""
        # The SEC's 'ix?doc=' viewer can break scrapers; we want the raw file.
        raw_url = report_url.replace("/ix?doc=", "")
        
        try:
            response = self.session.get(raw_url, headers=self.headers, timeout=15)
            response.raise_for_status() # Raise error for 4xx/5xx
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Download Error for {raw_url}: {e}")
            return ""

    def process_for_vector_db(self, html_content: str) -> List[Dict[str, str]]:
        """Parses HTML into chunks suitable for embeddings."""
        if not html_content:
            return []

        parser = sp.Edgar10QParser()
        
        # Suppress warnings for 10-K parsing using the 10-Q parser
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            elements = parser.parse(html_content)

        chunks = []
        for element in elements:
            text = getattr(element, "text", "").strip()
            
            # 150-200 chars is a good 'noise' floor for financial context
            if len(text) < 150:
                continue

            chunks.append({
                "text": text,
                "type": type(element).__name__,
            })
        return chunks
    
    def process_company_filings_task(self, symbol: str, db: Session):
        vector_service = VectorStoreService()
        
        # Discovery: Get metadata for recent filings
        meta_list = self.get_filing_metadata(symbol, years=1)
        
        for meta in meta_list:
            # Download
            html = self.download_filing_content(meta['reportUrl'])
            if not html:
                continue
                
            # Process and Save (uses your optimized batch/hash logic)
            vector_service.process_and_save_filing(
                db=db,
                symbol=symbol,
                html_content=html,
                metadata={
                    "filing_id": meta['accessNumber'],
                    "source_type": meta['form'],
                    "filed_date": meta['filedDate']
                }
            )