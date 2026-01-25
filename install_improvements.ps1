# install_improvements.ps1
# סקריפט אוטומטי להתקנת השיפורים

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ConTech Pro - התקנת שיפורים v2.1" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# עבור לתיקיית הפרויקט
$projectDir = "C:\Users\moshe\OneDrive\שולחן העבודה\ConTech"

if (-Not (Test-Path $projectDir)) {
    Write-Host "❌ שגיאה: תיקיית הפרויקט לא נמצאה!" -ForegroundColor Red
    Write-Host "נתיב: $projectDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "האם ליצור את התיקייה? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    
    if ($response -eq "Y" -or $response -eq "y") {
        New-Item -ItemType Directory -Path $projectDir -Force
        Write-Host "✅ תיקייה נוצרה" -ForegroundColor Green
    } else {
        Write-Host "מבטל..." -ForegroundColor Red
        exit
    }
}

Set-Location $projectDir
Write-Host "✅ עובד בתיקייה: $projectDir" -ForegroundColor Green
Write-Host ""

# ==========================================
# שלב 1: גיבוי
# ==========================================

Write-Host "📦 שלב 1: יוצר גיבוי..." -ForegroundColor Cyan

$backupDir = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

$filesToBackup = @("brain.py", "utils.py", "pages\manager.py")

foreach ($file in $filesToBackup) {
    if (Test-Path $file) {
        $destDir = Split-Path (Join-Path $backupDir $file)
        if (-Not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Copy-Item $file -Destination (Join-Path $backupDir $file) -Force
        Write-Host "  ✅ גיבוי: $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  קובץ לא נמצא: $file" -ForegroundColor Yellow
    }
}

Write-Host "✅ גיבוי הושלם ב: $backupDir" -ForegroundColor Green
Write-Host ""

# ==========================================
# שלב 2: בדיקת קבצים חדשים
# ==========================================

Write-Host "🔍 שלב 2: בודק קבצים חדשים..." -ForegroundColor Cyan

$newFiles = @(
    "brain_improved.py",
    "utils_improved.py", 
    "progress_utils.py",
    "db_monitor.py"
)

$filesExist = $true
foreach ($file in $newFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file קיים" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file חסר!" -ForegroundColor Red
        $filesExist = $false
    }
}

if (-Not $filesExist) {
    Write-Host ""
    Write-Host "❌ חסרים קבצים חדשים!" -ForegroundColor Red
    Write-Host "ודא שהורדת את הקבצים הבאים לתיקייה:" -ForegroundColor Yellow
    foreach ($file in $newFiles) {
        Write-Host "  - $file" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "האם להמשיך בכל זאת? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    
    if ($response -ne "Y" -and $response -ne "y") {
        Write-Host "מבטל..." -ForegroundColor Red
        exit
    }
}

Write-Host ""

# ==========================================
# שלב 3: החלפת קבצים
# ==========================================

Write-Host "🔄 שלב 3: מחליף קבצים..." -ForegroundColor Cyan

# אופציה להחלפה או שמירה
Write-Host "בחר אופציה:" -ForegroundColor Yellow
Write-Host "  1. שמור קבצים ישנים (brain_old.py, utils_old.py)" -ForegroundColor Cyan
Write-Host "  2. החלף ישירות (מעל הקבצים הישנים)" -ForegroundColor Cyan
Write-Host "  3. דלג על שלב זה" -ForegroundColor Cyan
$option = Read-Host "בחירה (1/2/3)"

switch ($option) {
    "1" {
        # שמור ישנים
        if (Test-Path "brain.py") {
            Rename-Item "brain.py" "brain_old.py" -Force
            Write-Host "  ✅ brain.py → brain_old.py" -ForegroundColor Green
        }
        
        if (Test-Path "utils.py") {
            Rename-Item "utils.py" "utils_old.py" -Force
            Write-Host "  ✅ utils.py → utils_old.py" -ForegroundColor Green
        }
        
        # העתק חדשים
        if (Test-Path "brain_improved.py") {
            Copy-Item "brain_improved.py" "brain.py" -Force
            Write-Host "  ✅ brain_improved.py → brain.py" -ForegroundColor Green
        }
        
        if (Test-Path "utils_improved.py") {
            Copy-Item "utils_improved.py" "utils.py" -Force
            Write-Host "  ✅ utils_improved.py → utils.py" -ForegroundColor Green
        }
    }
    
    "2" {
        # החלף ישירות
        if (Test-Path "brain_improved.py") {
            Copy-Item "brain_improved.py" "brain.py" -Force
            Write-Host "  ✅ brain.py הוחלף" -ForegroundColor Green
        }
        
        if (Test-Path "utils_improved.py") {
            Copy-Item "utils_improved.py" "utils.py" -Force
            Write-Host "  ✅ utils.py הוחלף" -ForegroundColor Green
        }
    }
    
    "3" {
        Write-Host "  ⏩ דילגתי על החלפת קבצים" -ForegroundColor Yellow
    }
    
    default {
        Write-Host "  ❌ בחירה לא תקינה, מדלג..." -ForegroundColor Red
    }
}

Write-Host ""

# ==========================================
# שלב 4: בדיקת תקינות
# ==========================================

Write-Host "🧪 שלב 4: בודק תקינות Python..." -ForegroundColor Cyan

# בדיקת syntax
$pythonFiles = @("brain.py", "utils.py", "progress_utils.py", "db_monitor.py")

foreach ($file in $pythonFiles) {
    if (Test-Path $file) {
        Write-Host "  🔍 בודק $file..." -ForegroundColor Gray
        $result = python -m py_compile $file 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    ✅ $file תקין" -ForegroundColor Green
        } else {
            Write-Host "    ❌ שגיאת syntax ב-$file" -ForegroundColor Red
            Write-Host "    $result" -ForegroundColor Yellow
        }
    }
}

Write-Host ""

# ==========================================
# שלב 5: Git
# ==========================================

Write-Host "📦 שלב 5: Git commit..." -ForegroundColor Cyan

# בדוק אם יש git
if (-Not (Test-Path ".git")) {
    Write-Host "⚠️  אין repository Git" -ForegroundColor Yellow
    Write-Host "האם לאתחל Git? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    
    if ($response -eq "Y" -or $response -eq "y") {
        git init
        Write-Host "  ✅ Git אותחל" -ForegroundColor Green
    } else {
        Write-Host "  ⏩ מדלג על Git" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  ✅ התקנה הושלמה!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        exit
    }
}

Write-Host "  🔍 בודק שינויים..." -ForegroundColor Gray
git status --short

Write-Host ""
Write-Host "האם ל-commit ו-push? (Y/N)" -ForegroundColor Yellow
$response = Read-Host

if ($response -eq "Y" -or $response -eq "y") {
    
    Write-Host "  📝 מוסיף קבצים..." -ForegroundColor Gray
    git add brain.py utils.py progress_utils.py db_monitor.py
    
    if (Test-Path "brain_improved.py") {
        git add brain_improved.py
    }
    if (Test-Path "utils_improved.py") {
        git add utils_improved.py
    }
    
    Write-Host "  💾 יוצר commit..." -ForegroundColor Gray
    git commit -m "Improvements v2.1: Enhanced legend analysis, error handling, and progress indicators

New Features:
- Few-shot learning for legend analysis (3 examples)
- Comprehensive error handling (3-layer protection)
- Visual progress indicators with context manager
- Database monitoring utility

Files:
- Added: brain_improved.py, utils_improved.py, progress_utils.py, db_monitor.py
- Updated: brain.py, utils.py

Technical:
- Temperature=0.3 for factual extraction
- Retry logic with 5 model fallbacks
- Input validation and cleanup
- Progress bar with auto-cleanup

Fixes:
- Better legend recognition (few-shot examples)
- Graceful degradation on errors
- User-friendly error messages"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Commit נוצר" -ForegroundColor Green
        
        Write-Host "  🚀 מעלה ל-GitHub..." -ForegroundColor Gray
        git push origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ הועלה ל-GitHub בהצלחה!" -ForegroundColor Green
        } else {
            Write-Host "  ❌ שגיאה בהעלאה" -ForegroundColor Red
            Write-Host "  💡 נסה: git push -u origin main" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ❌ שגיאה ב-commit" -ForegroundColor Red
    }
} else {
    Write-Host "  ⏩ מדלג על commit" -ForegroundColor Yellow
}

Write-Host ""

# ==========================================
# סיום
# ==========================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✅ התקנה הושלמה!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 סיכום:" -ForegroundColor Cyan
Write-Host "  ✅ גיבוי נוצר: $backupDir" -ForegroundColor Green
Write-Host "  ✅ קבצים עודכנו" -ForegroundColor Green
Write-Host "  ✅ בדיקת תקינות הושלמה" -ForegroundColor Green

Write-Host ""
Write-Host "🚀 צעדים הבאים:" -ForegroundColor Yellow
Write-Host "  1. הרץ: streamlit run app.py" -ForegroundColor White
Write-Host "  2. בדוק שהכל עובד" -ForegroundColor White
Write-Host "  3. Deploy ל-Cloud" -ForegroundColor White

Write-Host ""
Write-Host "📚 מידע נוסף:" -ForegroundColor Cyan
Write-Host "  - מדריך מלא: Complete_Installation_Guide.md" -ForegroundColor White
Write-Host "  - פתרון בעיות: ראה בסוף המדריך" -ForegroundColor White

Write-Host ""
Write-Host "בהצלחה! 🎉" -ForegroundColor Green
