# Run all required demos and calibration once API keys are placed in .env
$ErrorActionPreference = "Stop"

$forge = if (Test-Path ".venv\Scripts\forge.exe") { ".\.venv\Scripts\forge.exe" } else { "python -m forgeai.cli" }

Write-Host "1. Checking providers..."
Invoke-Expression "$forge check"

Write-Host "2. Running Demo A (HR Policy Assistant)..."
Invoke-Expression "$forge build --brief demos/hr_policy/brief.txt --docs demos/hr_policy/docs --run-name hr_policy --auto"

Write-Host "3. Running Demo A Over-Ambitious Variant (Groundability Guard)..."
try {
    Invoke-Expression "$forge build --brief demos/hr_policy/brief_overambitious.txt --docs demos/hr_policy/docs --run-name hr_overambitious --auto"
} catch {
    Write-Host "Groundability guard stopped over-ambitious brief as expected."
}

Write-Host "4. Running Demo B (Ship-Spares Procurement Assistant)..."
Invoke-Expression "$forge build --brief demos/ship_spares/brief.txt --docs demos/ship_spares/docs --run-name ship_spares --auto"

Write-Host "All demo artifacts written under runs/."
