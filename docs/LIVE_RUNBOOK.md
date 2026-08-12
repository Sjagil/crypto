# Live Runbook

Check current state:

```powershell
cd C:\Users\alhar\Documents\crypto
.\.venv\Scripts\python.exe .\main.py doctor
.\.venv\Scripts\python.exe .\main.py live reconcile
.\.venv\Scripts\python.exe .\main.py live status
.\.venv\Scripts\python.exe .\main.py telegram status
```

Start only when status shows no existing instance:

```powershell
.\.venv\Scripts\python.exe .\main.py autonomous-live start
```

Normal controls:

```powershell
.\.venv\Scripts\python.exe .\main.py autonomous-live pause
.\.venv\Scripts\python.exe .\main.py autonomous-live resume
.\.venv\Scripts\python.exe .\main.py autonomous-live shutdown
```

Emergency stop:

```powershell
.\.venv\Scripts\python.exe .\main.py live emergency-stop --reason "INCIDENT"
```

No signal means no order. It does not mean the service is inactive. Confirm
the live PID, heartbeat, private stream and reconciliation rather than
forcing an execution smoke trade into strategy accounting.
