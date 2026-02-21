# Railway logs dashboard guide

Use Railway’s **Observability** dashboard to view and filter your backend and frontend logs in one place.

## 1. Open the Observability dashboard

1. In [Railway](https://railway.com/dashboard), open your **project**.
2. Choose the right **environment** (e.g. Production).
3. Click **Observability** in the top project navigation.

## 2. Create your first dashboard

- If the dashboard is empty, click **“Start with a simple dashboard”** to get an auto-generated layout (spend, service metrics, and logs).
- Or click **“New”** (top right) to add widgets one by one.

## 3. Add log widgets

1. Click **“New”** (top right).
2. In the widget modal, set **Data source** to **Logs**.
3. Choose:
   - **Single service** (e.g. backend or frontend) or **multiple services**.
   - **Filter** (optional) to narrow what appears in that widget.

### Useful filters for your app

Railway normalizes logs; you can filter by level and by text in the message.

| Goal | Filter |
|------|--------|
| Errors only | `@level:error` |
| Warnings and errors | `@level:error OR @level:warn` |
| Request logs (backend) | `request` |
| Success / business events | `holding_created` or `plaid_token_exchanged` or `portfolio_summary` |
| Backend only (if multiple services) | Select the **backend service** as data source |
| Frontend only | Select the **frontend service** as data source |
| Client errors (from frontend → backend) | `[client]` |
| 5xx API errors | `API error 5` |
| Plaid | `plaid` or `Plaid` |
| Auth | `JWTError` or `login_success` or `auth_callback` |

Combine with **AND** / **OR** / **-** (negation), e.g.:

- `@level:error AND plaid`
- `request AND @httpStatus:>=500` (if HTTP logs are available)

## 4. Example dashboard layout

1. **Widget 1 – Errors**  
   Data source: Logs, service: backend.  
   Filter: `@level:error`  
   Name: e.g. “Backend errors”.

2. **Widget 2 – Warnings**  
   Data source: Logs, service: backend.  
   Filter: `@level:warn`  
   Name: e.g. “Backend warnings”.

3. **Widget 3 – Request traffic**  
   Data source: Logs, service: backend.  
   Filter: `request`  
   Name: e.g. “API requests”.

4. **Widget 4 – Success / business events**  
   Data source: Logs, service: backend.  
   Filter: `holding_created OR plaid_token_exchanged OR portfolio_summary`  
   Name: e.g. “Key success events”.

5. **Widget 5 – Frontend logs**  
   Data source: Logs, service: frontend.  
   No filter (or e.g. `@level:error` for frontend errors).  
   Name: e.g. “Frontend logs”.

6. **Widget 6 – Client errors (ingested by backend)**  
   Data source: Logs, service: backend.  
   Filter: `[client]`  
   Name: e.g. “Client-side errors”.

## 5. Arrange and save

- Click **Edit** (top right), then drag widgets to reorder and resize (bottom-right handle).
- Use the three-dot menu on a widget to edit its data source/filter or delete it.
- Click **Save** to keep the layout.

## 6. View logs in context

- In the Observability **Log explorer**, right-click a log line and choose **“View in context”** to see nearby lines (same request or timeframe).

## 7. Log retention and limits

- **Hobby / Trial:** 7 days retention.  
- **Pro:** 30 days retention; you can add **Monitors** (alerts) on metrics and use webhooks.
- Railway limits log throughput (e.g. 500 lines/sec per replica); if you hit it, reduce verbosity or sample logs.

## 8. Optional: structured JSON logs for better filtering

Your app currently logs plain text (`timestamp | level | name | message`). Railway still maps that to a `message` and infers `level` when possible.

To filter by **custom fields** (e.g. `user_id`, `symbol`) in the dashboard, you’d need to emit **one JSON object per line** from the backend, for example:

```json
{"level":"info","message":"holding_created","user_id":123,"symbol":"AAPL","holding_id":456}
```

Then in Railway you can use filters like `@user_id:123` or `@symbol:AAPL`. Implementing that would mean changing the Python logging formatter or adding a custom handler that prints JSON. The current format is fine for level- and text-based dashboards above.

## References

- [Railway – Viewing logs](https://docs.railway.com/guides/logs)
- [Railway – Observability dashboard](https://docs.railway.com/guides/observability)
