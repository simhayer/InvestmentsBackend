To start this project locally:
1. python3.12 -m venv venv
2. source venv/bin/activate 
3. pip install -r requirements.txt
4. Copy `.env.example` to `.env` and set variables (see below).
5. uvicorn main:app

Feedback form: the `/api/feedback/submit` endpoint requires `SUPABASE_PROJECT_URL` and `SUPABASE_SERVICE_ROLE_KEY` in `.env`. Get the service role key from Supabase: Project Settings → API → `service_role` (secret). Restart the backend after changing env.