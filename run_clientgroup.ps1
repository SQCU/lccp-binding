Write-Host "Launching 4 concurrent clients..."

$jobs = @(
Start-Job { python jsonprobe.py "The first prompt is about the history of Rome." }
Start-Job { python jsonprobe.py "Write a python function that calculates a fibonacci sequence." }
Start-Job { python jsonprobe.py "What is the capital of Mongolia? Explain its significance." }
Start-Job { python jsonprobe.py "The best thing about AI is its ability to learn." }
)

$jobs | Wait-Job
Write-Host "n--- Client Responses ---"
foreach ($job in $jobs) { 
Write-Host "--- Output from Job $($job.Id) ---" Receive-Job $job # This command prints the job's output to the console 
Write-Host "--------------------------n"
}
$jobs | Remove-Job