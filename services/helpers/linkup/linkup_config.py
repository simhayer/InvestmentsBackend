import os
from linkup import LinkupClient

LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
client = LinkupClient(api_key=LINKUP_API_KEY)