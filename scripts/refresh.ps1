param([switch]$SkipBuild, [switch]$SkipDeploy)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$logDirectory = Join-Path $projectRoot 'knowledge\cache'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
Start-Transcript -Path (Join-Path $logDirectory 'scheduled-update.log') -Append | Out-Null
try {
    & node 'node_modules/wrangler/bin/wrangler.js' whoami
    if ($LASTEXITCODE -ne 0) { throw 'Cloudflare authentication is unavailable.' }
    if (-not $SkipBuild) {
        & node 'scripts/discover.mjs'
        if ($LASTEXITCODE -ne 0) { throw 'Directory discovery failed; active corpus preserved.' }
        & node 'scripts/verify-sources.mjs'
        if ($LASTEXITCODE -ne 0) { throw 'Source verification failed; active corpus preserved.' }
        & 'venv/Scripts/python.exe' 'scripts/verify_supplements.py'
        if ($LASTEXITCODE -ne 0) { throw 'Official material notice verification failed.' }
        & 'venv/Scripts/python.exe' '-u' 'scripts/build_corpus.py'
        if ($LASTEXITCODE -ne 0) { throw 'Corpus build failed; active corpus preserved.' }
    }
    & node 'scripts/upload-corpus.mjs' '--bulk'
    if ($LASTEXITCODE -ne 0) { throw 'Corpus upload incomplete; next run will resume.' }
    if (-not $SkipDeploy) {
        & node 'scripts/publish-secrets.mjs'
        if ($LASTEXITCODE -ne 0) { throw 'Provider secret provisioning failed.' }
        & node 'node_modules/wrangler/bin/wrangler.js' deploy
        if ($LASTEXITCODE -ne 0) { throw 'Worker deployment failed.' }
    }
} finally {
    Stop-Transcript | Out-Null
}
