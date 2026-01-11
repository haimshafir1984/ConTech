# סקריפט להגדרת Git לפרויקט ConTech

Write-Host "=== הגדרת Git לפרויקט ConTech ===" -ForegroundColor Cyan

# עבור לתיקיית הפרויקט
$projectDir = "C:\Users\moshe\OneDrive\שולחן העבודה\ConTech"
Set-Location $projectDir
Write-Host "עובד בתיקייה: $projectDir" -ForegroundColor Green

# בדוק אם יש כבר .git בתיקיית הפרויקט
if (Test-Path ".git") {
    Write-Host "נמצא repository Git קיים. האם למחוק וליצור מחדש? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    if ($response -eq "Y" -or $response -eq "y") {
        Remove-Item -Recurse -Force .git
        Write-Host "מחקתי את ה-repository הקיים" -ForegroundColor Green
    } else {
        Write-Host "מבטל..." -ForegroundColor Red
        exit
    }
}

# אתחל Git repository חדש
Write-Host "`n1. מאתחל Git repository..." -ForegroundColor Cyan
git init
if ($LASTEXITCODE -ne 0) {
    Write-Host "שגיאה באתחול Git" -ForegroundColor Red
    exit
}

# הוסף את כל הקבצים
Write-Host "`n2. מוסיף קבצים..." -ForegroundColor Cyan
git add .
if ($LASTEXITCODE -ne 0) {
    Write-Host "שגיאה בהוספת קבצים" -ForegroundColor Red
    exit
}

# צור commit ראשון
Write-Host "`n3. יוצר commit ראשון..." -ForegroundColor Cyan
git commit -m "Initial commit: ConTech Pro - מערכת ניהול בנייה מקצועית"
if ($LASTEXITCODE -ne 0) {
    Write-Host "שגיאה ביצירת commit" -ForegroundColor Red
    exit
}

# שנה את שם ה-branch ל-main
Write-Host "`n4. משנה branch ל-main..." -ForegroundColor Cyan
git branch -M main

# שאל על ה-remote URL
Write-Host "`n5. הגדרת Remote Repository" -ForegroundColor Cyan
Write-Host "הכנס את ה-URL של ה-repository ב-GitHub:" -ForegroundColor Yellow
Write-Host "דוגמה: https://github.com/YOUR-USERNAME/ConTech.git" -ForegroundColor Gray
$remoteUrl = Read-Host "GitHub Repository URL"

if ($remoteUrl) {
    # בדוק אם יש כבר remote
    $existingRemote = git remote get-url origin 2>$null
    if ($existingRemote) {
        Write-Host "נמצא remote קיים. האם לעדכן? (Y/N)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -eq "Y" -or $response -eq "y") {
            git remote set-url origin $remoteUrl
        } else {
            Write-Host "מדלג על הגדרת remote" -ForegroundColor Gray
        }
    } else {
        git remote add origin $remoteUrl
    }
    
    Write-Host "`n6. מעלה ל-GitHub..." -ForegroundColor Cyan
    git push -u origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ הצלחה! הקוד הועלה ל-GitHub" -ForegroundColor Green
        Write-Host "Repository URL: $remoteUrl" -ForegroundColor Cyan
    } else {
        Write-Host "`n⚠️ שגיאה בעת העלאה. ודא ש:" -ForegroundColor Yellow
        Write-Host "  - ה-repository נוצר ב-GitHub" -ForegroundColor Gray
        Write-Host "  - יש לך הרשאות לכתוב ל-repository" -ForegroundColor Gray
        Write-Host "  - אתה מחובר ל-GitHub (git config --global user.name/user.email)" -ForegroundColor Gray
    }
} else {
    Write-Host "מדלג על העלאה. תוכל לעשות זאת מאוחר יותר עם:" -ForegroundColor Yellow
    Write-Host "  git remote add origin <URL>" -ForegroundColor Gray
    Write-Host "  git push -u origin main" -ForegroundColor Gray
}

Write-Host "`n=== סיום ===" -ForegroundColor Cyan
