import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
from services.finnhub.finnhub_service import FinnhubService
from services.filings.vector_store_service import VectorStoreService
from database import SessionLocal

class FilingService:
    def __init__(self):
        self.client = FinnhubService().get_finnhub_client()
        # Required User-Agent for SEC EDGAR access
        self.headers = {'User-Agent': 'WALLSTREETAI hayersimrat23@gmail.com'}
        self.session = requests.Session()

    def get_filing_metadata(self, symbol: str, years: int = 1) -> List[Dict[str, Any]]:
        """Fetch filtered 10-K and 10-Q metadata from Finnhub."""
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        
        try:
            all_filings = self.client.filings(symbol=symbol, _from=start_date)
            target_forms = {'10-K', '10-Q'}
            return [f for f in all_filings if f.get('form') in target_forms]
        except Exception as e:
            print(f"Finnhub Metadata Error: {e}")
            return []

    def download_filing_content(self, report_url: str) -> str:
        """Downloads raw HTML content, bypassing the iXBRL interactive viewer."""
        raw_url = report_url.replace("/ix?doc=", "")
        
        try:
            response = self.session.get(raw_url, headers=self.headers, timeout=20)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Download Error for {raw_url}: {e}")
            return ""

    def process_company_filings_task(self, symbol: str):
        db = SessionLocal()
        try:
            vector_service = VectorStoreService()
            meta_list = self.get_filing_metadata(symbol, years=1)

            total = 0
            for meta in meta_list:
                html = self.download_filing_content(meta.get("reportUrl", ""))
                if not html:
                    continue

                inserted = vector_service.process_and_save_filing(
                    db=db,
                    symbol=symbol,
                    html_content=html,
                    metadata={
                        "filing_id": meta.get("accessNumber"),
                        "form": meta.get("form"),
                        "filedDate": meta.get("filedDate"),
                        "report_url": meta.get("reportUrl"),
                    },
                )
                total += int(inserted or 0)

            db.commit()
            print(f"[filings] {symbol}: committed. inserted={total}")

        except Exception as e:
            db.rollback()
            print(f"[filings] {symbol}: FAILED -> {e}")
            raise
        finally:
            db.close()