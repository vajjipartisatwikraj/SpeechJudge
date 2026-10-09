# Test fixtures

`audio/*.wav` are speech clips generated with the Windows text-to-speech voices (David, Zira, Ravi and
Heera, the last two being Indian English). They are git-ignored and can be recreated at any time:

```powershell
pwsh tests/fixtures/make_audio.ps1      # PowerShell 7 (5.1 cannot select the en-IN voices)
```

Used by `tests/e2e` only (unit and integration tests synthesise their own audio in `tests/support.py`).
TTS speech is clean and regular, so it validates behaviour, not accuracy on real students.
