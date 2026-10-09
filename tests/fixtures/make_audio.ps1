# Regenerates the speech clips in tests/fixtures/audio/ with the Windows text-to-speech voices.
#   pwsh tests/fixtures/make_audio.ps1          (PowerShell 7; the 5.1 host cannot select the en-IN voices)
#
# Voices: David / Zira (US) and Ravi / Heera (Indian English). Output: 16 kHz, 16-bit, mono WAV.
# TTS speech is clean and regular: good for plumbing and relative-behaviour checks, NOT a stand-in
# for real student recordings when judging absolute accuracy.
Add-Type -AssemblyName System.Speech
$out = Join-Path $PSScriptRoot "audio"
New-Item -ItemType Directory -Force $out | Out-Null

$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

function Save-Clip($file, $text) {
  $synth.SetOutputToWaveFile((Join-Path $out $file), $fmt)
  $synth.Speak($text)
  $synth.SetOutputToNull()
}

# --- 1. pronunciation clips: 4 voices x 6 variants --------------------------------------------
$voices = [ordered]@{ david = "Microsoft David Desktop"; zira = "Microsoft Zira Desktop";
                      ravi = "Microsoft Ravi"; heera = "Microsoft Heera" }
$variants = [ordered]@{
  correct   = "The doctor suggested that the apple is good for your health."
  wrongword = "The doctor suggested that the apple is good for your wealth."
  mispron   = "The doktor sugjested dat the appel is gud for your helf."
  truncated = "The doctor suggested that the apple."
  thwords   = "I think these three thin thieves threw that tree."
  question  = "Did you finish the report before the meeting yesterday?"
}
foreach ($v in $voices.Keys) {
  try { $synth.SelectVoice($voices[$v]) } catch { Write-Warning "voice '$v' not installed, skipped"; continue }
  foreach ($c in $variants.Keys) { Save-Clip "$($v)_$($c).wav" $variants[$c] }
  Write-Host "voice $v done"
}

# --- 2. open-ended answers (default voice: David) ---------------------------------------------
$synth.SelectVoice($voices["david"]); $synth.Rate = 0
Save-Clip "reading.wav"      "The doctor suggested that the apple is good for your health."
Save-Clip "repeat_wrong.wav" "The doctor said that an apple is bad for your wealth."
Save-Clip "qa_good.wav"      "On weekends I usually wake up late. Then I go to the park with my friends and we play football. In the evening I watch a movie with my family."
Save-Clip "qa_offtopic.wav"  "Bananas are yellow. Trains are very fast and I like the color blue."
Save-Clip "story.wav"        "Last summer my family went to a small village near the mountains. On the first day we walked to a river and saw a little boy who had lost his dog. We helped him search the forest, and after two hours we found the dog sleeping under a tree. The boy was very happy, and his mother invited us for dinner. It was the best holiday I have ever had."
$synth.Dispose()
Write-Host "done: $((Get-ChildItem $out -Filter *.wav).Count) clips in $out"
