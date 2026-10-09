# Re-creates audio_01.wav ... audio_10.wav (about 10 s each) from benchmark_audio/manifest.json
# with the Windows text-to-speech voice "Microsoft David Desktop". Output: 16 kHz, 16-bit, mono WAV.
#
#   powershell -ExecutionPolicy Bypass -File benchmark_audio\make_benchmark_audio.ps1
#
# For every sentence the script tries a few speaking rates and keeps the one whose clip is closest to
# target_duration_s (10 s), so all clips are about the same length.
# The WAV files are committed, so this is only needed to change the sentences in manifest.json.
# TTS speech is clean and regular: right for timing benchmarks, NOT a stand-in for real student recordings.
Add-Type -AssemblyName System.Speech
$manifest = Get-Content (Join-Path $PSScriptRoot "manifest.json") -Raw | ConvertFrom-Json
$target = [double]$manifest.target_duration_s

$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("Microsoft David Desktop")

function Save-Clip($path, $text, $rate) {
  $synth.Rate = $rate
  $synth.SetOutputToWaveFile($path, $fmt)
  $synth.Speak($text)
  $synth.SetOutputToNull()
  return ((Get-Item $path).Length - 44) / 32000
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "bench_probe.wav"
foreach ($item in $manifest.items) {
  $best = 0; $bestGap = [double]::MaxValue
  foreach ($rate in 2, 1, 0, -1, -2, -3, -4) {
    $gap = [math]::Abs((Save-Clip $tmp $item.text $rate) - $target)
    if ($gap -lt $bestGap) { $bestGap = $gap; $best = $rate }
  }
  $seconds = Save-Clip (Join-Path $PSScriptRoot $item.file) $item.text $best
  "{0}: {1:N1} s (rate {2})" -f $item.file, $seconds, $best
}
Remove-Item $tmp -ErrorAction SilentlyContinue
$synth.Dispose()
