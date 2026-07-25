<#
.SYNOPSIS
    Quick launch script for Optuna hyperparameter optimization (PowerShell version)

.DESCRIPTION
    Runs hyperparameter sweeps for WARLOCK PPO with configurable modes.

.EXAMPLE
    .\launch_optuna.ps1 quick

.EXAMPLE
    .\launch_optuna.ps1 full

.EXAMPLE
    .\launch_optuna.ps1 resume warlock_full

.EXAMPLE
    .\launch_optuna.ps1 analyze warlock_full
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet('quick', 'full', 'resume', 'analyze', 'lstm', 'reward')]
    [string]$Mode,

    [Parameter(Position=1)]
    [string]$StudyName = "warlock_full"
)

$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition | Split-Path -Parent
Set-Location $PROJECT_DIR

switch ($Mode) {
    'quick' {
        Write-Host "=== QUICK SWEEP: 10 trials, 3 seeds, 50k steps ===" -ForegroundColor Green
        python -m src.agent.hpo_optuna `
            --n-trials 10 `
            --n-jobs 2 `
            --seeds "0,1,2" `
            --timesteps 50000 `
            --search-space "core+env" `
            --lambda-penalty 7.0 `
            --volatility-penalty 0.5 `
            --study-name "warlock_quick"
    }

    'full' {
        Write-Host "=== FULL SWEEP: 50 trials, 3 seeds, 100k steps ===" -ForegroundColor Green
        python -m src.agent.hpo_optuna `
            --n-trials 50 `
            --n-jobs 4 `
            --seeds "0,1,2" `
            --timesteps 100000 `
            --search-space "full" `
            --lambda-penalty 7.0 `
            --volatility-penalty 0.5 `
            --pruner "median" `
            --n-startup-trials 5 `
            --n-warmup-seeds 1 `
            --study-name "warlock_full"
    }

    'resume' {
        Write-Host "=== RESUME SWEEP: continuing '$StudyName' ===" -ForegroundColor Green
        python -m src.agent.hpo_optuna `
            --n-trials 50 `
            --n-jobs 4 `
            --seeds "0,1,2" `
            --timesteps 100000 `
            --search-space "full" `
            --study-name $StudyName
    }

    'analyze' {
        Write-Host "=== ANALYZE STUDY: '$StudyName' ===" -ForegroundColor Green
        python -m src.agent.hpo_optuna `
            --analyze `
            --study-name $StudyName `
            --show-top 20
    }

    'lstm' {
        Write-Host "=== LSTM ARCHITECTURE SWEEP ===" -ForegroundColor Green
        python -m src.agent.hpo_optuna `
            --n-trials 20 `
            --n-jobs 2 `
            --seeds "0,1,2" `
            --timesteps 50000 `
            --search-space "core+lstm" `
            --study-name "warlock_lstm"
    }

    'reward' {
        Write-Host "=== REWARD/ENV SWEEP ===" -ForegroundColor Green
        python -m src.agent.hpo_optuna `
            --n-trials 20 `
            --n-jobs 2 `
            --seeds "0,1,2" `
            --timesteps 50000 `
            --search-space "core+env" `
            --study-name "warlock_reward"
    }
}